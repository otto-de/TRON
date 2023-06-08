import itertools
import json
import random
import warnings
from copy import copy

import numpy as np
from torch import long, tensor
from torch.utils.data.dataset import IterableDataset

from src.shared.sample import (sample_in_batch_negatives, sample_uniform,
                               sample_uniform_negatives_with_shape)
from src.shared.utils import get_offsets


def label_session(session):
    without_label = session[:-1]
    labels = session[1:]
    for idx in range(len(without_label)):
        without_label[idx]['label'] = labels[idx]['aid']
    return without_label


def get_inactive_buffer_sessions(labeled_session_buffer):
    inactive_buffer_session_indices = []
    for session_idx, session in enumerate(labeled_session_buffer):
        if len(session) == 0:
            inactive_buffer_session_indices.append(session_idx)
    return inactive_buffer_session_indices


class Gru4RecDataset(IterableDataset):

    def __init__(self,
                 sessions_path,
                 total_sessions,
                 num_items,
                 max_seqlen,
                 shuffling_style="no_shuffling",
                 num_uniform_negatives=1,
                 num_in_batch_negatives=None,
                 reject_uniform_session_items=False,
                 reject_in_batch_items=True,
                 sampling_style="sessionwise",
                 batch_size=128):
        self.session_path = sessions_path
        self.total_sessions = total_sessions
        self.num_items = num_items
        self.max_seqlen = max_seqlen
        self.num_uniform_negatives = num_uniform_negatives
        self.num_in_batch_negatives = num_in_batch_negatives
        if self.num_in_batch_negatives is None:
            self.num_in_batch_negatives = batch_size - 1
        self.reject_uniform_session_items = reject_uniform_session_items
        self.reject_in_batch_items = reject_in_batch_items
        self.sampling_style = sampling_style
        self.shuffling_style = shuffling_style
        self.batch_size = batch_size
        self.line_offsets = get_offsets(sessions_path)
        self.__reset_dataset__()
        if self.sampling_style == "eventwise":
            self.sampling_style = "sessionwise"
            warnings.warn("Warning eventwise is not supported and is set to sessionwise ...")

    def __reset_dataset__(self):
        self.offset_queue = copy(self.line_offsets)
        if self.shuffling_style=="shuffle_without_replacement":
            random.shuffle(self.offset_queue)
        assert len(self.line_offsets) == self.total_sessions, f"{len(self.line_offsets)} != {self.total_sessions}"
        self.offset_queue = iter(self.offset_queue)
        self.labeled_session_buffer = [[]] * self.batch_size
        self.clicks = [[]] * self.batch_size

    def process_data(self, line_offsets):
        while True:
            keep_state = [1.] * self.batch_size
            with open(self.session_path, "rt") as f:
                inactive = get_inactive_buffer_sessions(self.labeled_session_buffer)
                for inactive_index in inactive:
                    try:
                        next_session_index = next(self.offset_queue)
                    except:
                        self.__reset_dataset__()
                        return
                    if self.shuffling_style=="shuffle_with_replacement":
                        next_session_index = line_offsets[np.random.randint(0, self.total_sessions)]
                    f.seek(next_session_index)
                    session = json.loads(f.readline())
                    self.labeled_session_buffer[inactive_index] = label_session(
                        session["events"][-(self.max_seqlen + 1):])
                    keep_state[inactive_index] = 0.
                    self.clicks[inactive_index] = [event['aid'] for event in
                                                   self.labeled_session_buffer[inactive_index]]
            batch = [session.pop(0) for session in self.labeled_session_buffer]
            clicks = [int(event["aid"]) for event in batch]
            labels = [int(event["label"]) for event in batch]
            if self.sampling_style == "batchwise":
                uniform_negatives = sample_uniform(self.num_items, [1, self.num_uniform_negatives],
                                                   set(itertools.chain.from_iterable(self.clicks)),
                                                   self.reject_uniform_session_items)
            else:
                uniform_negatives = np.array([sample_uniform_negatives_with_shape(session_clicks, self.num_items, 1,
                                                                                  self.num_uniform_negatives,
                                                                                  self.sampling_style,
                                                                                  self.reject_uniform_session_items) for
                                              session_clicks in
                                              self.clicks])
            in_batch_negatives = sample_in_batch_negatives(clicks, self.num_in_batch_negatives, [1] * self.batch_size,
                                                           self.reject_in_batch_items)
            yield {
                'clicks': tensor(clicks, dtype=long),
                'labels': tensor(labels, dtype=long).unsqueeze(1),
                'keep_state': tensor(keep_state).unsqueeze(1),
                'uniform_negatives': tensor(uniform_negatives, dtype=long),
                'in_batch_negatives': tensor(in_batch_negatives, dtype=long)
            }

    def __iter__(self):
        return self.process_data(self.line_offsets)

    def dynamic_collate(self, batch):
        batch = batch[0]
        return {
            'clicks': batch['clicks'],
            'labels': batch['labels'],
            'keep_state': batch['keep_state'],
            'uniform_negatives': batch['uniform_negatives'],
            'in_batch_negatives': batch['in_batch_negatives'],
            'mask': tensor([[1.] * self.batch_size])
        }

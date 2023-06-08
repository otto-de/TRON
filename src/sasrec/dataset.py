import json

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from src.shared.sample import (sample_in_batch_negatives, sample_uniform,
                               sample_uniform_negatives_with_shape)
from src.shared.utils import get_offsets


class SasRecDataset(Dataset):

    def __init__(self,
                 sessions_path,
                 total_sessions,
                 num_items,
                 max_seqlen,
                 num_uniform_negatives=1,
                 num_in_batch_negatives=0,
                 reject_uniform_session_items=False,
                 reject_in_batch_items=True,
                 sampling_style="eventwise",
                 shuffling_style="no_shuffling"
                 ):
        self.session_path = sessions_path
        self.total_sessions = total_sessions
        self.num_items = num_items
        self.max_seqlen = max_seqlen
        self.shuffling_style = shuffling_style
        self.num_uniform_negatives = num_uniform_negatives
        self.num_in_batch_negatives = num_in_batch_negatives
        self.reject_uniform_session_items = reject_uniform_session_items
        self.reject_in_batch_items = reject_in_batch_items
        self.sampling_style = sampling_style
        self.line_offsets = get_offsets(sessions_path)

        assert self.sampling_style in {"eventwise", "sessionwise", "batchwise"}
        assert len(self.line_offsets) == self.total_sessions, f"{len(self.line_offsets)} != {self.total_sessions}"

    def __len__(self):
        return self.total_sessions

    def __getitem__(self, idx):
        with open(self.session_path, "rt") as f:

            if self.shuffling_style=="shuffle_with_replacement":
                idx = np.random.randint(0,self.total_sessions)

            f.seek(self.line_offsets[idx])
            line = f.readline()
            session = json.loads(line)
            session = session["events"]

            assert sorted(session, key=lambda d: d["ts"]) == session

            clicks = [int(event["aid"]) for event in session if event["type"] == "clicks"]

            clicks = clicks[-(self.max_seqlen + 1):]
            session_len = min(len(clicks) - 1, self.max_seqlen)
            labels = clicks[1:]
            clicks = clicks[:-1]
            negatives = sample_uniform_negatives_with_shape(clicks, self.num_items, session_len, self.num_uniform_negatives, self.sampling_style, self.reject_uniform_session_items)

            return {'clicks': clicks, 'labels': labels, 'session_len': session_len, "uniform_negatives": negatives.tolist()}


    def dynamic_collate(self, batch):
        batch_clicks = list()
        batch_mask = list()
        batch_labels = list()
        batch_session_len = list()
        batch_positives = list()
        max_len = self.max_seqlen
        batch_uniform_negatives = list()
        in_batch_negatives = list()

        for item in batch:
            session_len = item["session_len"]
            batch_clicks.append((max_len - session_len) * [0] + item["clicks"])
            batch_mask.append((max_len - session_len) * [0.] + session_len * [1.])
            batch_labels.append((max_len - session_len) * [0] + item["labels"])
            batch_session_len.append(session_len)
            batch_positives.extend(item["clicks"])

            if self.sampling_style=="eventwise":
                batch_uniform_negatives.append((max_len - session_len) * [[0]*self.num_uniform_negatives] + item["uniform_negatives"]) 
            elif self.sampling_style=="sessionwise":
                batch_uniform_negatives.append(item["uniform_negatives"]) 
            
        if self.sampling_style=="batchwise":
            batch_uniform_negatives = sample_uniform(self.num_items, [self.num_uniform_negatives], set(batch_positives), self.reject_in_batch_items) 

        in_batch_negatives = sample_in_batch_negatives(batch_positives, self.num_in_batch_negatives, batch_session_len, self.reject_in_batch_items) 
        
        return {
            'clicks': torch.tensor(batch_clicks, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long), 
            'mask': torch.tensor(batch_mask, dtype=torch.float),
            'session_len': torch.tensor(batch_session_len, dtype=torch.long),
            'in_batch_negatives': torch.tensor(in_batch_negatives, dtype=torch.long), 
            'uniform_negatives': torch.tensor(batch_uniform_negatives, dtype=torch.long) 
        }

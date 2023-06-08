import itertools
from random import sample

import numpy as np


def _uniform_negatives(num_items, shape):
    return np.random.randint(1, num_items+1, shape)

def _uniform_negatives_session_rejected(num_items, shape, in_session_items):
        negatives = []
        for _ in range(np.prod(shape)):
            negative = np.random.randint(1, num_items+1)
            while negative in in_session_items:
                negative = np.random.randint(1, num_items+1)
            negatives.append(negative)
        return np.array(negatives).reshape(shape)

def _infer_shape(session_len, num_uniform_negatives, sampling_style):
    if sampling_style=="eventwise":
        return [session_len, num_uniform_negatives]
    elif sampling_style=="sessionwise":
        return [num_uniform_negatives]
    else:
         return []

def sample_uniform(num_items, shape, in_session_items, reject_session_items):
    if reject_session_items:
        return _uniform_negatives_session_rejected(num_items, shape, in_session_items)
    else: 
        return _uniform_negatives(num_items, shape)

def sample_uniform_negatives_with_shape(clicks, num_items, session_len, num_uniform_negatives, sampling_style, reject_session_items):
    in_session_items = set(clicks)
    shape = _infer_shape(session_len, num_uniform_negatives, sampling_style)
    if shape:
        negatives = sample_uniform(num_items, shape, in_session_items, reject_session_items)
    else: 
        negatives = np.array([])
    return negatives


def sample_in_batch_negatives(batch_positives, num_in_batch_negatives, batch_session_len, reject_session_items):
    in_batch_negatives = []
    positive_indices = itertools.accumulate(batch_session_len)
    positive_indices = [0] + [p for p in positive_indices]
    if reject_session_items:
        for i in range(len(positive_indices[:-1])):
            candidate_positives = batch_positives[:positive_indices[i]] + batch_positives[
                positive_indices[i + 1]:]
            in_batch_negatives.append(sample(candidate_positives, num_in_batch_negatives))
    else:
        for i in range(len(batch_session_len)):
            in_batch_negatives.append(sample(batch_positives, num_in_batch_negatives))
    return in_batch_negatives
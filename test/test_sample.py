import itertools

import numpy as np

from src.shared.sample import (_infer_shape, _uniform_negatives,
                               _uniform_negatives_session_rejected,
                               sample_in_batch_negatives, sample_uniform,
                               sample_uniform_negatives_with_shape)


def test_uniform_negatives():
    num_items = 10
    shape = [5,2]
    negatives = _uniform_negatives(num_items=num_items, shape=shape)
    assert negatives.shape == (5,2)
    assert set(list(itertools.chain(*negatives))).difference(set(range(1,11))) == set([])


def test_uniform_negatives_with_0():
    pass

def test_uniform_negatives_session_rejected():
    num_items = 10
    shape = [5,2]
    in_session_items = [1,5,10]

    negatives = _uniform_negatives_session_rejected(num_items=num_items, shape=shape, in_session_items=in_session_items)

    assert negatives.shape == (5,2)
    assert set(in_session_items).intersection(set(list(itertools.chain(*negatives.tolist())))) == set([])

def test_infer_shape():
    session_len = 5 
    num_uniform_negatives = 2
    shape_eventwise = _infer_shape(session_len=session_len, num_uniform_negatives=num_uniform_negatives, sampling_style="eventwise")
    shape_sessionwise = _infer_shape(session_len=session_len, num_uniform_negatives=num_uniform_negatives, sampling_style="sessionwise")
    shape_batchwise = _infer_shape(session_len=session_len, num_uniform_negatives=num_uniform_negatives, sampling_style="batchwise")

    assert shape_eventwise==[5,2]
    assert shape_sessionwise==[2,]
    assert shape_batchwise==[]

def test_sample_uniform():
    num_items = 10
    shape = [6,2]
    clicks = [7,4,3]
    with_rejection = sample_uniform(num_items=num_items, shape=shape, in_session_items=clicks, reject_session_items=True)
    without_rejection = sample_uniform(num_items=num_items, shape=shape, in_session_items=clicks, reject_session_items=False)

    assert with_rejection.shape == (6,2)
    assert without_rejection.shape == (6,2)

    for element in with_rejection.tolist():
        assert set(element).isdisjoint(set(clicks))

    for element in without_rejection.tolist():
        assert set(element).issubset(set(range(1,11)))

def test_sample_uniform_negatives_with_shape():
    clicks = [7,4,3]
    num_items = 10
    session_len = 12
    num_uniform_negatives = 3
    elementwise_negatives = sample_uniform_negatives_with_shape(clicks=clicks, num_items=num_items, session_len=session_len, num_uniform_negatives=num_uniform_negatives, sampling_style="eventwise", reject_session_items=False)
    sessionwise_negatives = sample_uniform_negatives_with_shape(clicks=clicks, num_items=num_items, session_len=session_len, num_uniform_negatives=num_uniform_negatives, sampling_style="sessionwise", reject_session_items=False)
    batchwise_negatives = sample_uniform_negatives_with_shape(clicks=clicks, num_items=num_items, session_len=session_len, num_uniform_negatives=num_uniform_negatives, sampling_style="batchwise", reject_session_items=False)
    
    assert elementwise_negatives.shape == (12,3)
    assert sessionwise_negatives.shape == (3,)
    assert batchwise_negatives.shape == (0,)

def test_sample_in_batch_negatives():
    batch_positives = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_in_batch_negatives = 2
    batch_session_len = [5,2,3]

    without_same_session_negatives = sample_in_batch_negatives(batch_positives=batch_positives, num_in_batch_negatives=num_in_batch_negatives, batch_session_len=batch_session_len, reject_session_items=True)
    with_same_session_negatives = sample_in_batch_negatives(batch_positives=batch_positives, num_in_batch_negatives=num_in_batch_negatives, batch_session_len=batch_session_len, reject_session_items=False)

    assert np.array(with_same_session_negatives).shape == (3, 2)
    for element in with_same_session_negatives:
        assert set(element).issubset(set(range(1,11)))

    assert np.array(without_same_session_negatives).shape == (3, 2)
    assert set(without_same_session_negatives[0]).issubset(set([6, 7, 8, 9, 10]))
    assert set(without_same_session_negatives[1]).issubset(set([1, 2, 3, 4, 5, 8, 9, 10]))
    assert set(without_same_session_negatives[2]).issubset(set([1, 2, 3, 4, 5, 6, 7]))
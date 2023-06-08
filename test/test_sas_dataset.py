import numpy as np
from torch import allclose, tensor
from torch.utils.data.dataloader import DataLoader

from src.sasrec.dataset import SasRecDataset


def test_dataset():
    session_path = "test/resources/train.jsonl"
    dataset = SasRecDataset(session_path, total_sessions=10, num_items=40_000, max_seqlen=6, shuffling_style="no_shuffling", num_uniform_negatives=3, num_in_batch_negatives=0, reject_uniform_session_items=False, sampling_style="eventwise")

    expected_first_session = {
        'clicks': [33838, 4759, 15406, 12887, 27601, 15406],
        'labels': [4759, 15406, 12887, 27601, 15406, 14564],
        'session_len': 6
    }

    expected_second_session = {'clicks': [36617], 'labels': [34257], 'session_len': 1}

    expected_third_session = {
        'clicks': [31292, 18083],
        'labels': [18083, 12957],
        'session_len': 2
    }

    expected_fourth_session = {
        'clicks': [14138],
        'labels': [8977],
        'session_len': 1
    }

    first_session = dataset.__getitem__(0)
    second_session = dataset.__getitem__(1)
    third_session = dataset.__getitem__(2)
    fourth_session = dataset.__getitem__(3)

    assert first_session['clicks'] == expected_first_session['clicks']
    assert first_session['labels'] == expected_first_session['labels']
    assert first_session['session_len'] == expected_first_session['session_len']
    assert np.array(first_session['uniform_negatives']).shape == (6, 3)

    assert second_session['clicks'] == expected_second_session['clicks']
    assert second_session['labels'] == expected_second_session['labels']
    assert second_session['session_len'] == expected_second_session['session_len']
    assert np.array(second_session['uniform_negatives']).shape == (1, 3)

    assert third_session['clicks'] == expected_third_session['clicks']
    assert third_session['labels'] == expected_third_session['labels']
    assert third_session['session_len'] == expected_third_session['session_len']
    assert np.array(third_session['uniform_negatives']).shape == (2, 3)

    assert fourth_session['clicks'] == expected_fourth_session['clicks']
    assert fourth_session['labels'] == expected_fourth_session['labels']
    assert fourth_session['session_len'] == expected_fourth_session['session_len']
    assert np.array(fourth_session['uniform_negatives']).shape == (1, 3)


def test_datalaoder():
    session_path = "test/resources/train.jsonl"
    dataset = SasRecDataset(sessions_path=session_path, total_sessions=10, num_items=40_000, max_seqlen=6, shuffling_style="no_shuffling", num_uniform_negatives=3, num_in_batch_negatives=2, reject_uniform_session_items=True, sampling_style="eventwise")
    dataloader = DataLoader(dataset,
                            batch_size=3,
                            shuffle=False,
                            collate_fn=dataset.dynamic_collate)



    expected_first_batch = {
        'clicks': tensor([
            [33838, 4759, 15406, 12887, 27601, 15406], 
            [0, 0, 0, 0, 0, 36617], 
            [0, 0, 0, 0, 31292, 18083]]),
        'labels': tensor([
            [4759, 15406, 12887, 27601, 15406, 14564], 
            [0, 0, 0, 0, 0, 34257], 
            [0, 0, 0, 0, 18083, 12957]]),
        'mask': tensor([
            [1., 1., 1., 1., 1., 1.], 
            [0., 0., 0., 0., 0., 1.], 
            [0., 0., 0., 0., 1., 1.],]),
        'session_len': tensor([6, 1, 2]),
    }

    for batch in dataloader:
        assert allclose(batch['clicks'],
                            expected_first_batch['clicks'])
        assert allclose(batch['labels'],
                            expected_first_batch['labels'])
        assert allclose(batch['mask'],
                            expected_first_batch['mask'])
        assert allclose(batch['session_len'],
                            expected_first_batch['session_len'])
        assert batch['in_batch_negatives'].shape == (3,2)
        assert batch['uniform_negatives'].shape == (3,6,3)
        assert set(batch['in_batch_negatives'].tolist()[0]).issubset([36617, 31292, 18083])
        assert set(batch['in_batch_negatives'].tolist()[1]).issubset([33838, 4759, 15406, 12887, 27601, 15406, 31292, 18083])
        assert set(batch['in_batch_negatives'].tolist()[2]).issubset([33838, 4759, 15406, 12887, 27601, 15406, 36617])
        break

    dataset.sampling_style="sessionwise"
    batch = next(iter(dataloader))
    assert batch['uniform_negatives'].shape == (3,3)

    dataset.sampling_style="batchwise"
    batch = next(iter(dataloader))
    assert batch['uniform_negatives'].shape == (3,)


def test_datalaoder_no_uniform_negatives():
    session_path = "test/resources/train.jsonl"
    dataset = SasRecDataset(sessions_path=session_path, total_sessions=10, num_items=40_000, max_seqlen=6, shuffling_style="no_shuffling", num_uniform_negatives=0, num_in_batch_negatives=2, reject_uniform_session_items=True, sampling_style="eventwise")
    dataloader = DataLoader(dataset,
                            batch_size=3,
                            shuffle=False,
                            collate_fn=dataset.dynamic_collate)

    for batch in dataloader:
        assert batch['uniform_negatives'].shape == (3,6,0)
        break

    dataset.sampling_style="sessionwise"
    batch = next(iter(dataloader))
    assert batch['uniform_negatives'].shape == (3,0)

    dataset.sampling_style="batchwise"
    batch = next(iter(dataloader))
    assert batch['uniform_negatives'].shape == (0,)


def test_datalaoder_no_in_batch_negatives():
    session_path = "test/resources/train.jsonl"
    dataset = SasRecDataset(sessions_path=session_path, total_sessions=10, num_items=40_000, max_seqlen=6, shuffling_style="no_shuffling", num_uniform_negatives=3, num_in_batch_negatives=0, reject_uniform_session_items=True, sampling_style="eventwise")
    dataloader = DataLoader(dataset,
                            batch_size=3,
                            shuffle=False,
                            collate_fn=dataset.dynamic_collate)


    for batch in dataloader:
        assert batch['in_batch_negatives'].shape == (3,0)
        break

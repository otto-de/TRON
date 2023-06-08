import torch
from torch import equal, tensor
from torch.utils.data.dataloader import DataLoader

from src.gru4rec.dataset import (Gru4RecDataset, get_inactive_buffer_sessions,
                                 label_session)


def test_label_session():
    session = [
        {'aid': 33838, 'ts': 1464127201.522, 'type': 'clicks'}, 
        {'aid': 4759, 'ts': 1464127218.472, 'type': 'clicks'}, 
        {'aid': 15406, 'ts': 1464127243.334, 'type': 'clicks'}, 
        {'aid': 12887, 'ts': 1464127245.905, 'type': 'clicks'}, 
        {'aid': 27601, 'ts': 1464127251.938, 'type': 'clicks'}, 
        {'aid': 15406, 'ts': 1464127265.936, 'type': 'clicks'},
        {'aid': 14564, 'ts': 1464127406.279, 'type': 'clicks'}
        ]

    expected = [
        {'aid': 33838, 'ts': 1464127201.522, 'type': 'clicks', 'label': 4759}, 
        {'aid': 4759, 'ts': 1464127218.472, 'type': 'clicks', 'label': 15406}, 
        {'aid': 15406, 'ts': 1464127243.334, 'type': 'clicks', 'label': 12887}, 
        {'aid': 12887, 'ts': 1464127245.905, 'type': 'clicks', 'label': 27601}, 
        {'aid': 27601, 'ts': 1464127251.938, 'type': 'clicks', 'label': 15406}, 
        {'aid': 15406, 'ts': 1464127265.936, 'type': 'clicks', 'label': 14564}
        ]
    
    assert label_session(session) == expected    


def test_get_inactive_buffer_sessions():
    labeled_session_buffer = [
        [], 
        [{'aid': 33838, 'ts': 1464127201.522, 'type': 'clicks', 'label': 4759}, {'aid': 4759, 'ts': 1464127218.472, 'type': 'clicks', 'label': 15406}],
        []
        ]
    expected = [0, 2]

    assert get_inactive_buffer_sessions(labeled_session_buffer) == expected


def test_dataset():
    session_path = "test/resources/train.jsonl"
    dataset = Gru4RecDataset(session_path, total_sessions=10,  num_items=40_000, max_seqlen=6, batch_size=3, shuffling_style="no_shuffling", sampling_style="eventwise", num_uniform_negatives=5, reject_uniform_session_items=True)

    expected_first_batch = {'clicks': tensor([33838, 36617, 31292]), 'labels': tensor([[4759], [34257], [18083]]), 'keep_state': tensor([[0.], [0.], [0.]])}
    expected_second_batch = {'clicks': tensor([4759, 14138, 18083]), 'labels': tensor([[15406], [8977], [12957]]), 'keep_state': tensor([[1.], [0.], [1.]])}
    first_batch = next(dataset.__iter__())

    assert equal(first_batch['clicks'], expected_first_batch['clicks'])
    assert equal(first_batch['labels'], expected_first_batch['labels'])
    assert equal(first_batch['keep_state'], expected_first_batch['keep_state'])
    assert first_batch['uniform_negatives'].shape == torch.Size([3, 5])
    assert first_batch['in_batch_negatives'].shape == torch.Size([3, 2])

    dataset.sampling_style = 'sessionwise'
    second_batch = next(dataset.__iter__())
    assert equal(second_batch['clicks'], expected_second_batch['clicks'])
    assert equal(second_batch['labels'], expected_second_batch['labels'])
    assert equal(second_batch['keep_state'], expected_second_batch['keep_state'])
    assert second_batch['uniform_negatives'].shape == torch.Size([3, 5])
    assert second_batch['in_batch_negatives'].shape == torch.Size([3, 2])

    dataset.sampling_style = 'batchwise'
    third_batch = next(dataset.__iter__())
    assert third_batch['uniform_negatives'].shape == torch.Size([1, 5])
    assert third_batch['in_batch_negatives'].shape == torch.Size([3, 2])


def test_datalaoder():
    session_path = "test/resources/train.jsonl"
    dataset =  Gru4RecDataset(session_path, total_sessions=10, num_items=40_000, max_seqlen=6, batch_size=3, shuffling_style="no_shuffling", sampling_style="eventwise")
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=dataset.dynamic_collate)

    expected = {'clicks': tensor([ 4869, 39930, 16618]), 'labels': tensor([[686], [537], [399]]), 'keep_state': tensor([[1.], [1.], [1.]])}

    for batch in dataloader:
        last_batch = batch
        assert True
    assert equal(last_batch['clicks'], expected['clicks'])
    assert equal(last_batch['labels'], expected['labels'])
    assert equal(last_batch['keep_state'], expected['keep_state'])
    assert len(last_batch['uniform_negatives'].tolist()) == 3
    assert len(last_batch['in_batch_negatives'].tolist()) == 3


def test_datalaoder_batchsize_too_large():
    session_path = "test/resources/train.jsonl"
    dataset =  Gru4RecDataset(session_path, total_sessions=10, num_items=40_000, max_seqlen=6, batch_size=11, shuffling_style="no_shuffling")
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=dataset.dynamic_collate)
    
    for batch in dataloader:
        assert False
    assert len(dataloader.dataset.labeled_session_buffer) == 11
    assert dataloader.dataset.labeled_session_buffer[-1] == []
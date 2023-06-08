import torch
from torch import tensor

from src.sasrec.model import SASRec

batch = {
    'clicks': tensor([
        [1, 2, 3, 4], 
        [0, 0, 0, 2], 
        [0, 0, 5, 6]]),
    'labels': tensor([
        [2, 3, 4, 5], 
        [0, 0, 0, 3], 
        [0, 0, 6, 7]]),
    'in_batch_negatives': tensor([
        [5, 6],
        [6, 4],
        [1, 2]
    ]),
    'uniform_negatives': tensor([
        [[5,6,7],[5,6,7],[5,6,7],[5,6,7]],
        [[4,5,6],[4,5,6],[4,5,6],[4,5,6]],
        [[3,4,9],[3,4,9],[3,4,9],[3,4,9]]
    ]),
    'mask': tensor([
        [1., 1., 1., 1.], 
        [0., 0., 0., 1.], 
        [0., 0., 1., 1.],]),
    'session_len': tensor([4, 1, 2]),
}

def test_forward():
    model = SASRec(
        hidden_size=8,
        dropout_rate=0.,
        max_len=3,
        num_items=16,
        learning_rate=0.01,
        batch_size=2,
        sampling_style='eventwise')

    item_indices = tensor([[2,5,6],[0,9,8]], dtype=torch.long)
    mask = tensor([[1.,1.,1.], [0.,1.,1.]], dtype=torch.float)
    
    actual_shape = model.forward(item_indices, mask).shape
    expected_shape = torch.Size([2, 3, 8])

    assert actual_shape == expected_shape

def test_forward_with_output_bias():
    model = SASRec(
        hidden_size=8,
        dropout_rate=0.,
        max_len=3,
        num_items=16,
        learning_rate=0.01,
        batch_size=2,
        output_bias=True,
        sampling_style='eventwise')

    item_indices = tensor([[2,5,6],[0,9,8]], dtype=torch.long)
    mask = tensor([[1.,1.,1.], [0.,1.,1.]], dtype=torch.float)
    
    actual_shape = model.forward(item_indices, mask).shape
    expected_shape = torch.Size([2, 3, 9])

    assert actual_shape == expected_shape



def test_training_step():
    model = SASRec(
        hidden_size=8,
        dropout_rate=0.,
        max_len=4,
        num_items=16,
        learning_rate=0.01,
        batch_size=2,
        sampling_style='eventwise')

    loss = model.training_step(batch, None)
    assert loss.shape == torch.Size([])


def test_training_step_not_shared_output_bias():
    model = SASRec(
        hidden_size=8,
        dropout_rate=0.,
        max_len=4,
        num_items=16,
        learning_rate=0.01,
        batch_size=3,
        output_bias=True,
        share_embeddings=False,
        sampling_style='eventwise')

    loss = model.training_step(batch, None)
    assert loss.shape == torch.Size([])

def test_training_step_not_shared_output_no_output_bias():
    model = SASRec(
        hidden_size=8,
        dropout_rate=0.,
        max_len=4,
        num_items=16,
        learning_rate=0.01,
        batch_size=3,
        output_bias=False,
        share_embeddings=False,
        sampling_style='eventwise')

    loss = model.training_step(batch, None)
    assert loss.shape == torch.Size([])

def test_validation_step():
    model = SASRec(
        hidden_size=8,
        dropout_rate=0.,
        max_len=4,
        num_items=16,
        learning_rate=0.01,
        batch_size=3,
        sampling_style='eventwise')

    model.validation_step(batch, None)
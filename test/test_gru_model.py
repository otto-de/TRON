import torch
from torch import allclose, equal, tensor

from src.gru4rec.model import GRU4REC, clean_state

batch = {
        'clicks': tensor([1, 2]),
        'labels': tensor([[2], [3]]),
        'in_batch_negatives': tensor([
            [[5, 6]],
            [[6, 4]]
        ]),
        'uniform_negatives': tensor([
            [[5,6,7]],
            [[4,5,6]]
        ]),
        'keep_state': tensor([
            [1.], [1.]
        ]),
        'mask': tensor([
            [1., 1.],
        ]),
    }

def test_clean_state():
    curr_state = torch.ones(2, 3, 4)
    keep_state = tensor([[1.], [0.], [1.]])

    expected = tensor([
        [[1.,1.,1.,1.],[0.,0.,0.,0.],[1.,1.,1.,1.]],
        [[1.,1.,1.,1.],[0.,0.,0.,0.],[1.,1.,1.,1.]]
    ])
    assert equal(clean_state(curr_state, keep_state), expected)


def test_gru4Rec():
    model = GRU4REC(num_items=40_000,hidden_size=10,num_layers=2,batch_size=3, dropout_rate=0.)
    click_indices = tensor([33838, 33838, 33838])
    in_state = torch.ones(2, 3, 10) # num_layer, batch_size, hidden_dim
    keep_state = tensor([[1.], [0.], [1.]])

    gru_output, out_state = model.forward(click_indices, in_state, keep_state)

    assert gru_output.shape == torch.Size([3, 1, 10])
    assert out_state.shape == torch.Size([2, 3, 10])
    assert not allclose(gru_output[0], gru_output[1])
    assert allclose(gru_output[0], gru_output[2])


def test_gru4Re_with_output_bias():
    model = GRU4REC(num_items=40_000,hidden_size=10,num_layers=2,batch_size=3, dropout_rate=0., output_bias=True)
    click_indices = tensor([33838, 33838, 33838])
    in_state = torch.ones(2, 3, 10) # num_layer, batch_size, hidden_dim
    keep_state = tensor([[1.], [0.], [1.]])

    gru_output, out_state = model.forward(click_indices, in_state, keep_state)

    assert gru_output.shape == torch.Size([3, 1, 11])
    assert out_state.shape == torch.Size([2, 3, 10])
    assert not allclose(gru_output[0], gru_output[1])
    assert allclose(gru_output[0], gru_output[2])


def test_training_step_shared_no_bias():
    model = GRU4REC(num_items=40_000,hidden_size=10,num_layers=2,batch_size=2, dropout_rate=0.)

    loss = model.training_step(batch, None)
    assert loss.shape == torch.Size([])
    assert not allclose(model.current_state, torch.zeros(2,2,10)) 


def test_training_step_not_shared_no_bias():
    model = GRU4REC(num_items=40_000,hidden_size=10,num_layers=2,batch_size=2, dropout_rate=0., output_bias=False, share_embeddings=False)

    loss = model.training_step(batch, None)
    assert loss.shape == torch.Size([])
    assert not allclose(model.current_state, torch.zeros(2,2,10)) 

def test_training_step_not_shared_bias():
    model = GRU4REC(num_items=40_000,hidden_size=10,num_layers=2,batch_size=2, dropout_rate=0., output_bias=True, share_embeddings=False)

    loss = model.training_step(batch, None)
    assert loss.shape == torch.Size([])
    assert not allclose(model.current_state, torch.zeros(2,2,10))


def test_training_step_not_shared_bias():
    model = GRU4REC(num_items=40_000,hidden_size=10,num_layers=2,batch_size=2, dropout_rate=0., output_bias=True, share_embeddings=False, original_gru=True)

    loss = model.training_step(batch, None)
    assert loss.shape == torch.Size([])
    assert not allclose(model.current_state, torch.zeros(2,2,10)) 

def test_validation_step():
    model = GRU4REC(num_items=40_000,hidden_size=10,num_layers=2,batch_size=2, dropout_rate=0.)
    model.validation_step(batch, None)
    assert True

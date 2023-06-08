import torch
from torch import sigmoid, softmax, tensor

from src.shared.loss import (_bpr_max_loss_regularization,
                             _bpr_max_loss_unregulized, _diff_logits,
                             _elementwise_bpr_max_loss_per_negative,
                             _elementwise_sampled_softmax_loss, bce_loss,
                             bpr_max_loss, sampled_softmax_loss)


def test_elementwise_sampled_softmax_loss():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], [
                              [0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]], dtype=torch.float)
    mask = tensor([[1., 1., 1., 1., 1.], [0., 1., 1., 1., 1.]])
    target = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    expected_elementwise_loss = tensor([[2.4938, 4.4197, 6.4093, 4.1488, 4.1488],[0.0000, 6.4093, 4.4197, 2.4938, 4.1488]])
    assert torch.allclose(expected_elementwise_loss, _elementwise_sampled_softmax_loss(pos_logits, neg_logits, mask, target), atol=1e-5)


def test_sampled_softmax_loss():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], [
                              [0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]], dtype=torch.float)
    mask = tensor([[1., 1., 1., 1., 1.], [0., 1., 1., 1., 1.]])
    expected_loss = tensor(39.0920) / 9.
    assert torch.allclose(expected_loss, sampled_softmax_loss(pos_logits, neg_logits, mask))
    

def test_binary_cross_entropy_loss():
    mask = tensor([[1., 1.], [1., 0.]])
    positive_logits = tensor([[[1.], [-2.]],[[3.], [4.]]])
    negative_logits = tensor([[[1.1, 1.2], [1.2, 2.1]],[[-1., 4.], [1.,- 4.]]])
    assert torch.allclose(tensor(2.6397), bce_loss(positive_logits, negative_logits, mask), atol=0.001)


def test_difference_positive_and_negative_logits():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], 
                        [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], 
                        [[0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]], dtype=torch.float)
   
    expected_diff = tensor([[[0, -1, -2], [-2, -3, -4], [-4, -5, -6], [-2, 1, -4], [-2, 1, -4]],
                           [[0, 0, 0], [-6, -5, -4], [-4, -3, -2], [-2, -1, 0], [-2, 1, -4]]], dtype=torch.float)

    assert torch.equal(_diff_logits(pos_logits, neg_logits), expected_diff)


def test_elementwise_bpr_max_loss_per_negative():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], 
                        [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], 
                        [[0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]], dtype=torch.float)
    expected = tensor([[[0.0450, 0.0658, 0.0793], [0.0107, 0.0116, 0.0120], [0.0016, 0.0016, 0.0016], [0.0141, 0.0043, 0.0157], [0.0141, 0.0043, 0.0157]],
                       [[0.1667, 0.1667, 0.1667], [0.0016, 0.0016, 0.0016], [0.0120, 0.0116, 0.0107], [0.0793, 0.0658, 0.0450], [0.0141, 0.0043, 0.0157]]])
    actual = _elementwise_bpr_max_loss_per_negative(pos_logits, neg_logits)

    assert torch.allclose(actual, expected, atol=0.0001)

    
def test_elementwise_bpr_max_loss():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], 
                        [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], 
                        [[0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]], dtype=torch.float)
    mask = tensor([[1., 1., 1., 1., 1.],
                   [0., 1., 1., 1., 1.]])

    expected_unmasked = tensor([[1.6602, 3.3726, 5.3391, 3.3785, 3.3785],
                                [0.6929, 5.3391, 3.3726, 1.6602, 3.3785]], dtype=torch.float)
    actual_bpr_max_loss_unregulized_unmasked, actual_bpr_max_loss_unregulized = _bpr_max_loss_unregulized(pos_logits, neg_logits, mask)

    assert torch.allclose(actual_bpr_max_loss_unregulized_unmasked, expected_unmasked, atol=0.1)
    assert torch.allclose(actual_bpr_max_loss_unregulized, tensor(30.8793 / 9), atol=0.01)


def test_bpr_max_loss_regularization():
    penalty = 1.
    neg_logits = tensor([[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                         [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]]], dtype=torch.float)
    mask = tensor([[1., 1., 1., 1., 1.],
                   [0., 1., 1., 1., 1.]])

    expected_regularization = tensor(349 / 9)
    actual_regularization = _bpr_max_loss_regularization(neg_logits, penalty, mask)

    assert torch.allclose(actual_regularization, expected_regularization)


def test_bpr_max_loss():
    pos_logits = tensor([[[1], [2], [3], [4], [4]], 
                        [[0], [3], [2], [1], [4]]], dtype=torch.float)
    neg_logits = tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [6, 3, 8], [6, 3, 8]], 
                        [[0, 0, 0], [9, 8, 7], [6, 5, 4], [3, 2, 1], [6, 3, 8]]], dtype=torch.float)
    mask = tensor([[1., 1., 1., 1., 1.],
                   [0., 1., 1., 1., 1.]])
    penalty = 0.
    
    assert torch.allclose(bpr_max_loss(penalty, pos_logits, neg_logits, mask), tensor(30.8793 / 9), atol=0.01)
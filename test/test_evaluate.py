import torch
from torch import allclose, equal, tensor

from src.shared.evaluate import (calculate_ranks, mean_metric, pointwise_mrr,
                                 pointwise_recall,
                                 validate_batch_per_timestamp)


def test_pointwise_recall():
    ranks = tensor([[1., 0., float('inf')],
                    [0., 1., 1.],
                    [2., 2., 1.]])

    mask = tensor([[1., 1., 1.],
                   [1., 1., 0.],
                   [0., 0., 0.]])

    cutoffs = tensor([1, 3]).int()

    expected = tensor([
        [
            [0., 1., 0.],  # k=1
            [1., 0., 0.],  # k=1
            [0., 0., 0.]   # k=1
        ],
        [
            [1., 1., 0.],  # k=3
            [1., 1., 0.],  # k=3
            [0., 0., 0.]   # k=3
        ]
    ])
    assert allclose(pointwise_recall(ranks, cutoffs, mask), expected)


def test_pointwise_recall_per_timestamp():
    logits = tensor([[5., 6., 7., 1.],
                     [5., 6., 7., 1.],
                     [5., 6., 7., 1.]])

    labels = tensor([1, 2, 0])

    mask = tensor([1., 1., 0.])

    cutoffs = tensor([1, 3]).int()

    expected = tensor([[[0., 1., 0.]],  # k=1
                       [[1., 1., 0.]]])  # k=3
    ranks = calculate_ranks(logits, labels, cutoffs)
    assert allclose(pointwise_recall(ranks, cutoffs, mask), expected)


def test_mean_recall():
    recall_matrix = tensor([
        [
            [0., 0., 1.],  # k=1
            [0., 0., 0.],  # k=1
        ],
        [
            [0., 0., 0.],  # k=3
            [1., 1., 0.],  # k=3
        ]
    ])

    mask = tensor([[1., 1., 1.],
                   [1., 1., 0.]])

    expected = tensor([
        ((0 + 0 + 1) + (0 + 0 + 0)) / 5,  # k=1
        ((0 + 0 + 0) + (0 + 1 + 1)) / 5   # k=3
    ])
    assert allclose(mean_metric(recall_matrix, mask), expected)


def test_calculate_ranks():
    logits = tensor([[[5., 6., 7., 1.], [1., 2., 3., 0.], [4., 5., 1., 0.]],
                     [[5., 6., 7., 1.], [1., 2., 3., 0.], [4., 5., 1., 0.]],
                     [[5., 6., 7., 1.], [1., 2., 3., 0.], [4., 5., 1., 0.]]])

    labels = tensor([[1, 2, 3],
                     [2, 1, 0],
                     [0, 0, 0]])

    cutoffs = tensor([1, 3]).int()

    expected_ranks = tensor([[1., 0., float('inf')],
                             [0., 1., 1.],
                             [2., 2., 1.]])

    assert equal(calculate_ranks(logits, labels, cutoffs), expected_ranks)


def test_pointwise_mrr():
    ranks = tensor([[1., 0., float('inf')],
                    [0., 1., 1.],
                    [2., 2., 1.]])

    mask = tensor([[1., 1., 1.],
                   [1., 1., 0.],
                   [0., 0., 0.]])

    cutoffs = tensor([1, 3]).int()

    expected = tensor([[[0., 1., 0.],
                        [1., 0., 0.],
                        [0., 0., 0.]],

                       [[.5, 1., 0.],
                        [1., .5, 0.],
                        [0., 0., 0.]]])
    assert equal(pointwise_mrr(ranks, cutoffs, mask), expected)


def test_mean_mrr():
    mrr_matrix = tensor([[[0., 1., 0.],
                        [1., 0., 0.],
                        [0., 0., 0.]],

                       [[.5, 1., 0.],
                        [1., .5, 0.],
                        [0., 0., 0.]]])

    mask = tensor([[1., 1., 1.],
                   [1., 1., 0.],
                   [0., 0., 0.]])

    expected = tensor([
        ((0 + 1 + 0) + (1 + 0 + 0) + (0 + 0 + 0)) / 5,  # k=1
        ((.5 + 1 + 0) + (1 + .5 + 0) + (0 + 0 + 0)) / 5   # k=3
    ])

    assert equal(mean_metric(mrr_matrix, mask), expected)

def test_validate_batch_per_timestamp():
    x_hat = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                           [0.1, 0.2, 0.3, 0.4],
                           [0.1, 0.2, 0.3, 0.4]],
                          [[0.1, 0.2, 0.3, 0.4],
                           [0.1, 0.2, 0.3, 0.4],
                           [0.1, 0.2, 0.3, 0.4]]])
    output_embedding = torch.nn.Embedding(100, 4)

    batch = {
        'labels': torch.tensor([[1, 2, 0],
                                [1, 2, 0]]),
        'mask': torch.tensor([[0., 1., 1.],
                              [1., 1., 1.]])
    }
    cut_offs = torch.tensor([1, 3]).int()

    recalls, mrrs = validate_batch_per_timestamp(batch, x_hat, output_embedding, cut_offs)

    assert recalls.shape == torch.Size([2])
    assert mrrs.shape == torch.Size([2])
    assert torch.greater_equal(recalls, 0).all()
    assert torch.less_equal(recalls, 1).all()
    assert torch.greater_equal(mrrs, 0).all()
    assert torch.less_equal(mrrs, 1).all()

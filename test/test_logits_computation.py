import torch
from torch import equal, tensor
from torch.nn import Embedding

from src.shared.logits_computation import (lookup_and_multiply,
                                           multiply_head_with_embedding)


def test_multiply_head_with_embedding_batchwise():
    transformer_head = tensor([[[1.,0.],[0., 1.],[0.,0.]], [[1.,1.], [2., 0.],[0.,2.]]]) # [2,3,2] (batch_size, sequence_length, embedding_size)
    batchwise_negative_embedding = tensor([[1.,0.], [1., 1.]]) # [2,2] (negative_samples, embedding_size)
    expected_batchwise_multiplication = tensor([[[1. , 1.], [0., 1.], [0.,0.]], [[1., 2.], [2. , 2.], [0., 2.]]]) # [2,3,2] (batch_size, sequence_length, negative_samples)
    assert equal(multiply_head_with_embedding(transformer_head,batchwise_negative_embedding), expected_batchwise_multiplication)


def test_multiply_head_with_embedding_sessionwise():
    transformer_head = tensor([[[1.,0.],[0., 1.],[0.,0.]], [[1.,1.], [2., 0.],[0.,2.]]]) # [2,3,2] (batch_size, sequence_length, embedding_size)
    sessionwise_negative_embedding = tensor([[[1.,0.], [1., 1.]],[[1.,1.], [2., -1.]]]) # [2,2,2] (batch_size, negative_samples, embedding_size)
    expected_sessionwise_multiplication = tensor([[[1. , 1.], [0., 1.], [0.,0.]], [[2., 1.], [2. , 4.], [2., -2.]]]) # [2,3,2] (batch_size, sequence_length, negative_samples)
    assert equal(multiply_head_with_embedding(transformer_head,sessionwise_negative_embedding), expected_sessionwise_multiplication)


def test_multiply_head_with_embedding_eventwise():
    transformer_head = tensor([[[1.,0.],[0., 1.],[0.,0.]], [[1.,1.], [2., 0.],[0.,2.]]]).unsqueeze(-2) # [2,3,1,2] (batch_size, sequence_length, 1, embedding_size)
    eventwise_negative_embedding = tensor([[[[1.,0.], [1., 1.]],[[0., 1.], [1., 0.]], [[0.,0.], [0., 0.]]],[[[1.,1.], [2., 2.]], [[2., 0.], [3., 0.]],[[0.,1.], [0., 3.]]]]) # [2,3,2,2] (batch_size, sequence_length, negative_samples, embedding_size)
    expected_eventwise_multiplication = tensor([[[1. , 1.], [1., 0.], [0.,0.]], [[2., 4.], [4. , 6.], [2., 6.]]])  # [2,3,2] (batch_size, sequence_length, negative_samples)
    assert equal(multiply_head_with_embedding(transformer_head,eventwise_negative_embedding).squeeze(-2), expected_eventwise_multiplication)


def test_multiply_head_with_embedding_positives():
    transformer_head = tensor([[[1.,0.],[0., 1.],[0.,0.]], [[1.,1.], [2., 0.],[0.,2.]]]).unsqueeze(-2)  # [2,3,1,2] (batch_size, sequence_length, embedding_size)
    eventwise_positive_embedding = tensor([[[1.,0.], [0., 1.], [0.,0.]],[[1.,1.], [2., 0.], [0.,1.]]]).unsqueeze(-2) # [2,3,1,2] (batch_size, sequence_length, embedding_size)
    expected_positive_multiplication = tensor([[1., 1., 0.,], [2., 4., 2.]])  # [2,3] (batch_size, sequence_length)
    assert equal(multiply_head_with_embedding(transformer_head,eventwise_positive_embedding).squeeze(-1).squeeze(-1), expected_positive_multiplication)


def test_lookup_and_multiply_eventwise():
    transformer_head = tensor([[[1.,0.],[0., 1.],[0.,0.]], [[1.,1.], [2., 0.],[0.,2.]]])
    embedding_layer = Embedding(20, 2, padding_idx=0)
    positives = tensor([[2,5,6],[7,9,8]], dtype=torch.long)
    eventwise_uniform_negatives = tensor([[[1, 2],[3, 4], [5, 6]],[[7, 8], [9, 10],[11, 12]]], dtype=torch.long) 
    in_batch_negatives = tensor([[1, 3, 4],[6, 5, 8]], dtype=torch.long)

    expected_pos_logits_shape = [2,3,1]
    expected_neg_logits_shape = [2,3,5]
    actual_pos_logits, actual_neg_logits = lookup_and_multiply(transformer_head, positives, eventwise_uniform_negatives, in_batch_negatives, embedding_layer, 'eventwise')

    assert actual_pos_logits.shape == torch.Size(expected_pos_logits_shape)
    assert actual_neg_logits.shape == torch.Size(expected_neg_logits_shape)


def test_lookup_and_multiply_no_uniform_negatives():
    transformer_head = tensor([[[1.,0.],[0., 1.],[0.,0.]], [[1.,1.], [2., 0.],[0.,2.]]])
    embedding_layer = Embedding(20, 2, padding_idx=0)
    positives = tensor([[2,5,6],[7,9,8]], dtype=torch.long) #(batch_size, seqlen)
    elementwise_uniform_negatives = tensor([[[],[], []],[[], [],[]]], dtype=torch.long) 
    in_batch_negatives = tensor([[1, 3, 4],[6, 5, 8]], dtype=torch.long)
    
    expected_pos_logits_shape = [2,3,1]
    expected_neg_logits_shape = [2,3,3]
    actual_pos_logits, actual_neg_logits = lookup_and_multiply(transformer_head, positives, elementwise_uniform_negatives, in_batch_negatives, embedding_layer, 'eventwise')

    assert actual_pos_logits.shape == torch.Size(expected_pos_logits_shape)
    assert actual_neg_logits.shape == torch.Size(expected_neg_logits_shape)


def test_lookup_and_multiply_no_in_batch_negatives():
    transformer_head = tensor([[[1.,0.],[0., 1.],[0.,0.]], [[1.,1.], [2., 0.],[0.,2.]]])
    embedding_layer = Embedding(20, 2, padding_idx=0)
    positives = tensor([[2,5,6],[7,9,8]], dtype=torch.long)
    elementwise_uniform_negatives = tensor([[[1, 2],[3, 4], [5, 6]],[[7, 8], [9, 10],[11, 12]]], dtype=torch.long) 
    in_batch_negatives = tensor([[],[]], dtype=torch.long)

    expected_pos_logits_shape = [2,3,1]
    expected_neg_logits_shape = [2,3,2]
    actual_pos_logits, actual_neg_logits = lookup_and_multiply(transformer_head, positives, elementwise_uniform_negatives, in_batch_negatives, embedding_layer, 'eventwise')

    assert actual_pos_logits.shape == torch.Size(expected_pos_logits_shape)
    assert actual_neg_logits.shape == torch.Size(expected_neg_logits_shape)


def test_multiply_transformerhead_with_candidates_per_timestamp():
    transformer_head = tensor([[1.,0.], [1.,1.]])
    positive_embedding = tensor([[.2, .4],[.1, .2]])
    expected_multiplication = tensor([[.2, .1], [.6, .3]])

    assert equal(multiply_head_with_embedding(transformer_head, positive_embedding), expected_multiplication)

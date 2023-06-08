import torch
from torch import cat, exp, log, sigmoid, softmax, sum, tensor
from torch.nn import CrossEntropyLoss

from src.shared.logits_computation import lookup_and_multiply

ce_loss = CrossEntropyLoss(reduction="none")


def _elementwise_sampled_softmax_loss(pos_logits, neg_logits, mask, target):
    sm_logits = cat((pos_logits, neg_logits), dim=-1)
    shape = sm_logits.shape
    return ce_loss(sm_logits.reshape([-1, shape[-1]]), target).reshape([shape[0], shape[1]]) * mask


def sampled_softmax_loss(pos_logits, neg_logits, mask, device="cpu"):
    target = tensor([0], device=device).tile(mask.numel())
    elementwise_ssm_loss = _elementwise_sampled_softmax_loss(pos_logits, neg_logits, mask, target)
    return sum(elementwise_ssm_loss) / sum(mask)


def bce_loss(pos_logits, neg_logits, mask, epsilon=1e-10, device="cpu"):
    loss = log(1. + exp(-pos_logits) + epsilon) + log(1. + exp(neg_logits) + epsilon).mean(-1, keepdim=True)
    return (loss * mask.unsqueeze(-1)).sum() / mask.sum()


def _diff_logits(pos_logits, neg_logits):
    return (pos_logits - neg_logits)


def _elementwise_bpr_max_loss_per_negative(pos_logits, neg_logits):
    logits_diff = sigmoid(_diff_logits(pos_logits, neg_logits))
    s_j = softmax(neg_logits - torch.max(neg_logits, dim=-1)[0].unsqueeze(-1), dim=-1)
    return s_j * logits_diff


def _bpr_max_loss_unregulized(pos_logits, neg_logits, mask):
    bpr_max_loss_per_element = -log(sum(_elementwise_bpr_max_loss_per_negative(pos_logits, neg_logits), dim=-1))
    return bpr_max_loss_per_element, sum(bpr_max_loss_per_element * mask) / sum(mask)


def _bpr_max_loss_regularization(neg_logits, penalty, mask):
    regularization = penalty * sum(softmax(neg_logits, dim=-1) * neg_logits * neg_logits, dim=-1)
    return sum(regularization * mask) / sum(mask)


def bpr_max_loss(penalty, pos_logits, neg_logits, mask, device="cpu"):
    _, unregulized_bpr_max_loss = _bpr_max_loss_unregulized(pos_logits, neg_logits, mask)
    return unregulized_bpr_max_loss + _bpr_max_loss_regularization(neg_logits, penalty, mask)


def calc_loss(loss_fn,
              x_hat,
              labels,
              uniform_negatives,
              in_batch_negatives,
              mask,
              embeddings,
              sampling_style,
              final_activation,
              topk_sampling=False,
              topk_sampling_k=1000,
              device="cpu"):
    pos_logits, neg_logits = lookup_and_multiply(x_hat, labels, uniform_negatives, in_batch_negatives, embeddings,
                                                 sampling_style)
    if topk_sampling:
        neg_logits, _ = torch.topk(neg_logits, k=topk_sampling_k, dim=-1)
    pos_scores, neg_scores = final_activation(pos_logits), final_activation(neg_logits)
    return loss_fn(pos_scores, neg_scores, mask, device=device)

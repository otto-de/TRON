from torch import concat


def multiply_head_with_embedding(prediction_head, embeddings):
    return prediction_head.matmul(embeddings.transpose(-1, -2))


def lookup_and_multiply(prediction_head, positives, uniform_negatives, in_batch_negatives, embedding_layer, sampling_style):
    positive_logits = multiply_head_with_embedding(prediction_head.unsqueeze(-2),
                                                   embedding_layer(positives).unsqueeze(-2)).squeeze(-1)

    if sampling_style == "eventwise":
        uniform_negative_logits = multiply_head_with_embedding(prediction_head.unsqueeze(-2),
                                                               embedding_layer(uniform_negatives)).squeeze(-2)
    else:
        uniform_negative_logits = multiply_head_with_embedding(prediction_head, embedding_layer(uniform_negatives))

    in_batch_negative_logits = multiply_head_with_embedding(prediction_head, embedding_layer(in_batch_negatives))
    negative_logits = concat([uniform_negative_logits, in_batch_negative_logits], dim=-1)
    return positive_logits, negative_logits

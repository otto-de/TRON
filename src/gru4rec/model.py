import math
import warnings
from functools import partial

import pytorch_lightning as pl
import torch
from torch import concat, nn, tensor

from src.shared.evaluate import validate_batch_per_timestamp
from src.shared.loss import (bce_loss, bpr_max_loss, calc_loss,
                             sampled_softmax_loss)


def sparse_output(item_lookup, bias_lookup, output, items_to_predict):
    embeddings = item_lookup(items_to_predict)
    logits = torch.matmul(embeddings, output.t())
    bias = bias_lookup(items_to_predict).squeeze(1)
    return bias + logits.t()


def dense_output(linear_layer, output, items_to_predict):
    return linear_layer(output)[:, items_to_predict.view(-1)]


def clean_state(curr_state, keep_state):
    return curr_state * keep_state

class GRU4REC(pl.LightningModule):

    def __init__(self,
                 hidden_size,
                 dropout_rate,
                 num_items,
                 batch_size,
                 sampling_style="batchwise",
                 topk_sampling=False,
                 topk_sampling_k=1000,
                 learning_rate=0.001,
                 num_layers=1,
                 loss='bce',
                 bpr_penalty=None,
                 optimizer='adagrad',
                 output_bias=False,
                 share_embeddings=True,
                 original_gru=False,
                 final_activation=True):
        super(GRU4REC, self).__init__()
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_rate
        self.batch_size = batch_size
        self.sampling_style = sampling_style
        if sampling_style == "eventwise":
            warnings.warn("Warning eventwise is not supported and is set to sessionwise ...")
            self.sampling_style = sampling_style
        self.output_bias = output_bias
        self.share_embeddings = share_embeddings
        self.original_gru = original_gru

        if original_gru:
            warnings.warn("Warning gru original cannot share input and output embeddings, share embedding is set to False")
            self.share_embeddings = False

        if output_bias and share_embeddings:
            self.item_embedding = nn.Embedding(num_items + 1, hidden_size + 1, padding_idx=0)
        elif self.original_gru:
            self.item_embedding = nn.Embedding(num_items + 1, 3 * hidden_size, padding_idx=0)
        else:
            self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)

        if share_embeddings:
            self.output_embedding = self.item_embedding
        elif (not share_embeddings) and output_bias:
            self.output_embedding = nn.Embedding(num_items + 1, hidden_size + 1, padding_idx=0)
        else:
            self.output_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)

        torch.nn.init.xavier_uniform_(self.item_embedding.weight.data, gain=1 / math.sqrt(6))
        torch.nn.init.xavier_uniform_(self.output_embedding.weight.data, gain=1 / math.sqrt(6))

        self.gru = nn.GRU(int(3 * self.hidden_size) if self.original_gru else self.hidden_size,
                          self.hidden_size,
                          self.num_layers,
                          dropout=self.dropout_hidden,
                          batch_first=True)
        if final_activation:
            self.final_activation = nn.ELU(0.5)
        else:
            self.final_activation = nn.Identity()

        if self.original_gru:
            self.gru.weight_ih_l0 = nn.Parameter(data=torch.eye(3 * self.hidden_size), requires_grad=False)
        self.register_buffer('current_state', torch.zeros([num_layers, batch_size, hidden_size], device=self.device))
        self.register_buffer('loss_mask', torch.ones(1, self.batch_size, device=self.device))
        self.register_buffer('bias_ones', torch.ones([self.batch_size, 1, 1]))
        self.loss_fn = loss
        if self.loss_fn == 'bce':
            self.loss = bce_loss
        elif self.loss_fn == 'ssm':
            self.loss = sampled_softmax_loss
        elif self.loss_fn == 'bpr-max':
            if bpr_penalty is not None:
                self.loss = partial(bpr_max_loss, bpr_penalty)
            else:
                raise ValueError('bpr_penalty must be provided for bpr_max loss')
        else:
            raise ValueError('Loss function not supported')

        self.topk_sampling = topk_sampling
        self.topk_sampling_k = topk_sampling_k
        self.optimizer = optimizer
        self.save_hyperparameters()

    def forward(self, item_indices, in_state, keep_state):
        embedded = self.item_embedding(item_indices.unsqueeze(1))
        embedded = embedded[:, :, :-1] if self.output_bias and self.share_embeddings else embedded
        in_state = clean_state(in_state, keep_state)
        gru_output, out_state = self.gru(embedded, in_state)
        scores = concat([gru_output, self.bias_ones], dim=-1) if self.output_bias else gru_output
        return scores, out_state

    def training_step(self, batch, _):
        x_hat, c_state = self.forward(batch["clicks"], self.current_state, batch["keep_state"])

        self.current_state = c_state.detach()
        train_loss = calc_loss(self.loss, x_hat, batch["labels"], batch["uniform_negatives"], batch["in_batch_negatives"],
                               batch["mask"], self.output_embedding, self.sampling_style, self.final_activation,
                               self.topk_sampling, self.topk_sampling_k, self.device)

        self.log("train_loss", train_loss)

        return train_loss

    def validation_step(self, batch, _batch_idx):
        x_hat, self.current_state = self.forward(batch["clicks"], self.current_state, batch["keep_state"])
        cut_offs = tensor([5, 10, 20], device=self.device)
        recall, mrr = validate_batch_per_timestamp(batch, x_hat, self.output_embedding, cut_offs)
        test_loss = calc_loss(self.loss, x_hat, batch["labels"], batch["uniform_negatives"], batch["in_batch_negatives"],
                              batch["mask"], self.output_embedding, self.sampling_style, self.final_activation,
                              self.topk_sampling, self.topk_sampling_k, self.device)
        for i, k in enumerate(cut_offs.tolist()):
            self.log(f'recall_cutoff_{k}', recall[i])
            self.log(f'mrr_cutoff_{k}', mrr[i])
        self.log('test_seq_len', x_hat.shape[1])
        self.log('test_loss', test_loss)

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('Optimizer not supported, please use adam or adagrad')
        return optimizer

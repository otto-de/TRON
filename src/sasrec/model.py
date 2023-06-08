from functools import partial
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import concat, diag, logical_and, logical_or, nn, tensor, tile
from torch.nn import Dropout

from src.shared.evaluate import validate_batch_per_timestamp
from src.shared.logits_computation import multiply_head_with_embedding
from src.shared.loss import (bce_loss, bpr_max_loss, calc_loss,
                             sampled_softmax_loss)


class DynamicPositionEmbedding(torch.nn.Module):

    def __init__(self, max_len, dimension):
        super(DynamicPositionEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, dimension)
        self.pos_indices = torch.arange(0, self.max_len, dtype=torch.int)
        self.register_buffer('pos_indices_const', self.pos_indices)

    def forward(self, x, device='cpu'):
        seq_len = x.shape[1]
        return self.embedding(self.pos_indices_const[-seq_len:]) + x


class SASRec(pl.LightningModule):

    def __init__(self,
                 hidden_size,
                 dropout_rate,
                 max_len,
                 num_items,
                 batch_size,
                 sampling_style,
                 topk_sampling=False,
                 topk_sampling_k=1000,
                 learning_rate=0.001,
                 num_layers=2,
                 loss='bce',
                 bpr_penalty=None,
                 optimizer='adam',
                 output_bias=False,
                 share_embeddings=True,
                 final_activation=False):
        super(SASRec, self).__init__()
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_items = num_items
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.output_bias = output_bias
        self.share_embeddings = share_embeddings
        self.future_mask = torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1)
        self.register_buffer('future_mask_const', self.future_mask)
        self.register_buffer('seq_diag_const', ~diag(torch.ones(max_len, dtype=torch.bool)))
        self.register_buffer('bias_ones', torch.ones([self.batch_size, self.max_len, 1]))
        if output_bias and share_embeddings:
            self.item_embedding = nn.Embedding(num_items + 1, hidden_size + 1, padding_idx=0)
        else:
            self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.positional_embedding_layer = DynamicPositionEmbedding(max_len, hidden_size)

        torch.nn.init.xavier_uniform_(self.item_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.positional_embedding_layer.embedding.weight.data)

        if share_embeddings:
            self.output_embedding = self.item_embedding
        elif (not share_embeddings) and output_bias:
            self.output_embedding = nn.Embedding(num_items + 1, hidden_size + 1, padding_idx=0)
        else:
            self.output_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)

        self.norm = nn.LayerNorm([hidden_size])
        self.input_dropout = Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=1,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout_rate,
                                                   batch_first=True,
                                                   norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers, norm=self.norm)
        self.merge_attn_mask = True
        if final_activation:
            self.final_activation = nn.ELU(0.5)
        else:
            self.final_activation = nn.Identity()

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

        self.sampling_style = sampling_style
        self.topk_sampling = topk_sampling
        self.topk_sampling_k = topk_sampling_k
        self.optimizer = optimizer
        self.save_hyperparameters()

    def merge_attn_masks(self, padding_mask):
        batch_size = padding_mask.shape[0]
        seq_len = padding_mask.shape[1]

        if not self.merge_attn_mask:
            return self.future_mask_const[:seq_len, :seq_len]

        padding_mask_broadcast = ~padding_mask.bool().unsqueeze(1)
        future_masks = tile(self.future_mask_const[:seq_len, :seq_len], (batch_size, 1, 1))
        merged_masks = logical_or(padding_mask_broadcast, future_masks)
        # Always allow self-attention to prevent NaN loss
        # See: https://github.com/pytorch/pytorch/issues/41508
        diag_masks = tile(self.seq_diag_const[:seq_len, :seq_len], (batch_size, 1, 1))
        return logical_and(diag_masks, merged_masks)

    def forward(self, item_indices, mask):
        att_mask = self.merge_attn_masks(mask)
        items = self.item_embedding(
            item_indices)[:, :, :-1] if self.output_bias and self.share_embeddings else self.item_embedding(item_indices)
        x = items * np.sqrt(self.hidden_size)
        x = self.positional_embedding_layer(x)
        x = self.encoder(self.input_dropout(x), att_mask)
        return concat([x, self.bias_ones], dim=-1) if self.output_bias else x

    def training_step(self, batch, _):
        x_hat = self.forward(batch["clicks"], batch["mask"])
        train_loss = calc_loss(self.loss, x_hat, batch["labels"], batch["uniform_negatives"], batch["in_batch_negatives"],
                               batch["mask"], self.output_embedding, self.sampling_style, self.final_activation,
                               self.topk_sampling, self.topk_sampling_k, self.device)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, _batch_idx):
        x_hat = self.forward(batch['clicks'], batch['mask'])
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

    def export_topk_onnx(self, out_dir):
        top_k_model = TopKModel(self)
        top_k_model.export_onnx(out_dir)

    def export(self, out_dir):
        self.export_topk_onnx(out_dir)


class TopKModel(pl.LightningModule):

    def __init__(self, model: SASRec):
        super(TopKModel, self).__init__()
        self.model = model
        # example input for self.forward(item_indices, k)
        self.example_input_array = (torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]), torch.tensor(10))

    def forward(self, item_indices, k):
        mask = torch.ones(item_indices.shape[0]).unsqueeze(0)
        self.model.merge_attn_mask = False
        x_hat = self.model.forward(item_indices.unsqueeze(0), mask)[:, -1]
        logits = multiply_head_with_embedding(x_hat, self.model.item_embedding.weight)
        logits[0][0] = -torch.inf  # set score for padding item to -inf
        scores, indices = torch.topk(logits, k)
        return indices.squeeze(0), scores.squeeze(0)

    def export_onnx(self, out_dir, verbose=True):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.to_onnx(f"{out_dir}/sasrec.onnx",
                     export_params=True,
                     opset_version=13,
                     verbose=verbose,
                     do_constant_folding=False,
                     input_names=["item_indices", "k"],
                     output_names=[f"indices", "scores"],
                     dynamic_axes={
                         'item_indices': {
                             0: 'sequence'
                         },
                         'indices': {
                             0: 'k'
                         },
                         'scores': {
                             0: 'k'
                         }
                     })
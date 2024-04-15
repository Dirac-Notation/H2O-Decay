import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.opt.modeling_opt import OPTAttention


__all__ = ['convert_kvcache_opt_heavy_recent', 'OPTAttention_Mask']


def local_heavy_hitter_mask(attn_weights, heavy_budget, penalty):

    # attn_weights (head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]

    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    penalty_factor = torch.arange(heavy_budget,0,-1).unsqueeze(1).to(dtype_attn_weights).to(attn_weights.device) - 1
    penalty_factor = penalty**penalty_factor

    penaltied_attn = tmp_attn[:,:heavy_budget,:] * penalty_factor
    accumulated_attention_score = torch.sum(penaltied_attn, dim=-2) #(head, keys)
    accumulated_attention_score[:, heavy_budget:] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom[:, :heavy_budget, :heavy_budget] = True

    for token_index in range(heavy_budget, seq_length):
        attn_row = attn_weights[:,token_index,:]
        previous_mask = mask_bottom[:, token_index-1, :]
        attn_row = attn_row * previous_mask + ~previous_mask * torch.finfo(attn_weights.dtype).min

        tmp_attn_index = nn.functional.softmax(attn_row, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

        accumulated_attention_score *= penalty
        accumulated_attention_score += tmp_attn_index

        _, tmp_topk_index = accumulated_attention_score.topk(k=heavy_budget-1, dim=-1)

        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        mask_bottom_index[:, token_index] = True

        mask_bottom[:,token_index,:] = mask_bottom_index
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index

    mask_bottom = torch.tril(mask_bottom, diagonal=0)
    
    return mask_bottom, accumulated_attention_score

class OPTAttention_Mask(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        heavy_ratio: float,
        recent_ratio: float,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        penalty = 1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.heavy_budget_ratio = heavy_ratio
        self.recent_budget_ratio = recent_ratio
        self.attention_masks_next = None 
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

        self.penalty = penalty

    def _reset_masks(self):
        self.attention_masks_next = None 
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if (self.previous_scores is None):
            self.heavy_budget = int(self.heavy_budget_ratio * attn_weights.shape[-1])
            self.recent_budget = int(self.recent_budget_ratio * attn_weights.shape[-1])
            self.cache_budget = self.heavy_budget + self.recent_budget

            if self.heavy_budget > 0:
                mask_bottom, self.previous_scores = local_heavy_hitter_mask(attn_weights, self.heavy_budget, self.penalty) # Default: No padding applied to input
            else:
                mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
            
            # Recent Mask
            ones = torch.ones_like(attn_weights, dtype=torch.bool)
            ones = torch.triu(ones, diagonal=-self.recent_budget)
            mask_bottom = torch.logical_or(mask_bottom, ones)

            # Combine h2o+recent and apply casual mask
            mask_bottom = torch.tril(mask_bottom, diagonal=0)
            # mask_bottom = ones
            
            attn_weights[~mask_bottom] = torch.min(attention_mask)

            if attn_weights.dtype == torch.float16:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
            else:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            
            self.attention_masks_next = torch.ones(attn_weights.shape[0], 1, attn_weights.shape[2]+1).to(attn_weights.dtype).to(attn_weights.device)
            self.attention_masks_next[:,:,:-1] = mask_bottom[:,-1,:].unsqueeze(1)
            
        else:
            if self.attention_masks_next is not None:
                attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min
            
            # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
            if attn_weights.dtype == torch.float16:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
            else:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            # attn_weights (heads, q-tokens, k-tokens) -> q 방향으로 합치기
            # Scoring
            if attn_weights.shape[1] > 1:
                penalty = torch.arange(attn_weights.shape[1],0,-1).unsqueeze(1).to(attn_weights.dtype).to(attn_weights.device) - 1
                penalty = self.penalty**penalty
                current_scores_sum = attn_weights*penalty
                current_scores_sum = current_scores_sum.sum(1)

            if not self.previous_scores == None:
                current_scores_sum[:, :-1] += self.penalty*self.previous_scores #(Enlarge Sequence)

            self.previous_scores = current_scores_sum #(heads, k-tokens)
            attn_mask = torch.zeros(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(attn_weights.dtype).to(attn_weights.device)

            attn_tokens_all = self.previous_scores.shape[-1]
            if attn_tokens_all > self.cache_budget:
                # activate most recent k-cache
                if self.recent_budget > 1:
                    attn_mask[:, -self.recent_budget:] = 1
                    selected_set = self.previous_scores[:, :-self.recent_budget+1]
                else:
                    # activate historical best self.cache_budget - self.recent_budget tokens.
                    # self.previous_scores # (k-Cache - 1)
                    attn_mask[:, -1] = 1
                    selected_set = self.previous_scores[:, :]

                if not self.heavy_budget == 0:
                    _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                    attn_mask = attn_mask.scatter(-1, keep_topk, 1)

            self.attention_masks_next = attn_mask.unsqueeze(1)

            score_mask = attn_mask[:,:-1]
            # score_mask[:, -self.recent_budget:] = 1
            self.previous_scores = self.previous_scores * score_mask

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        # torch.set_printoptions(sci_mode=False, profile="full")
        # print(attn_mask[0])

        return attn_output, attn_weights_reshaped, past_key_value


def convert_kvcache_opt_heavy_recent(model, config):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_opt_heavy_recent(module, config)

        if isinstance(module, OPTAttention) or isinstance(module, OPTAttention_Mask):
            model._modules[name] = OPTAttention_Mask(
                embed_dim=module.embed_dim,
                num_heads=config.num_attention_heads,
                heavy_ratio = config.heavy_ratio,
                recent_ratio = config.recent_ratio,
                dropout=config.attention_dropout,
                is_decoder=True,
                bias=config.enable_bias,
                penalty=config.penalty
            )
    return model
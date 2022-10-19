#!/usr/bin/env python3
#####################################
#
# Filename : ferral_ent.py
#
# Projectname :
#
# Author : Oskar Taubert
#
# Creation Date : Mon 20 Jan 2020 03:26:21 PM CET
#
# Last Modified : Wed 01 Apr 2020 07:49:14 PM CEST
#
# ####################################

import torch
from torch import nn

from ..layers import OuterConcatLayer, OuterSumLayer, OuterProductLayer
from ..layers import build_mlp_list


class AttentionLayer(nn.Module):
    ''' Established softmax attention with position-wise two layer MLP afterwards '''
    def __init__(self, d, nheads, dropout=0.0, d_ff=None, attention_module='default'):
        """
        d : embedding dimension for the attention module
        nheads : number of heads for multi head attention module
        dropout : dropout probability for all dropout modules in this block
        d_ff : dimension of the feed foward layer, if None is set to 4*d
        attention_module : 'default' is the torch.nn.MultiheadAttention module, 'custom' is the super special BaumBauenAttentionModule
        """
        super(AttentionLayer, self).__init__()
        if d_ff is None:
            d_ff = 4 * d

        if attention_module == 'default':
            attn_type = nn.MultiheadAttention
        elif attention_module == 'custom':
            attn_type = BaumBauenAttentionModule

        self.attn = attn_type(d, nheads, dropout)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.ff = nn.Sequential(*[nn.Linear(d, d_ff), nn.LeakyReLU(0.1), nn.Dropout(p=dropout), nn.Linear(d_ff, d)])

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        return

    def forward(self, x):
        '''
        Input: (l, b, c) = (leaves, batch, features)
        Output: (l,b, c)
        '''
        # Here attention is the matrix of attention weights
        result, attention = self.attn(x, x, x)  # (l, b, c), (b, l, l)
        result = x + self.dropout1(result)  # (l, b, c)
        result = self.norm1(result)  # (l, b, c)
        result = result + self.dropout2(self.ff(result))  # (l, b, c)
        result = self.norm2(result)  # (l, b, c)

        return result


class BridgeLayer(nn.Module):
    '''
    Bridging layer between single and pair-wise representation using outer concatenation.
    This is for converting a representation of the leaves into an LCA matrix
    '''
    def __init__(self, method='sum', format='NCHW'):
        super(BridgeLayer, self).__init__()
        self.format = format

        supported = ['sum', 'product', 'concat']
        assert method in supported, f'method must be one of: {supported}'
        self.method = method

        if method == 'sum':
            self.expand = OuterSumLayer()
        elif method == 'product':
            self.expand = OuterProductLayer()
        elif method == 'concat':
            self.expand = OuterConcatLayer()

    def forward(self, x):
        '''
        Input of shape (l, b, d)
        Output of shape (b, d, l, l) if format is NCHW
        Output of shape (b, l, l, d) if format is NHWC
        Output of shape (l*l, b, d) if format is LNC
        For our purposes, dims 3 and 4 should be symmetric, so we don't care a lot about their order.
        Note: output dimesnion d is actually 2*d if concat method selected
        '''
        x = self.expand(x)  # (l, l, b, d), if concat: (l, l, b, 2d)
        l, _, b, d = x.size()

        # Covers future cases with 3 output dims
        # if len(self.format) == 4:
        #     x = x.reshape(l, l, b, d)
        if len(self.format) == 3:
            x = x.reshape(l * l, b, d)

        if self.format == 'NCHW':
            return x.permute(2, 3, 0, 1)  # (b, d, l, l)
        elif self.format == 'NHWC':
            return x.permute(2, 0, 1, 3)  # (b, l, l, d)
        elif self.format == 'LNC':
            return x
        else:
            raise


class BaumBauenAttentionModule(nn.Module):
    '''
    Custom attention module that uses a pairwise MLP instead of outer product and softmax to generate the attention map
    '''
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(BaumBauenAttentionModule, self).__init__()
        self.embed_dim = embed_dim
        # self.kdim = kdim if kdim is not None else embed_dim
        # self.vdim = vdim if kdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.attn_mlp = nn.Sequential(*build_mlp_list(2 * embed_dim, n_mlp=4, dim_mlp=embed_dim, out_feats=num_heads, dropout=dropout))
        self.v_mlp = nn.Sequential(*build_mlp_list(embed_dim, n_mlp=3, dim_mlp=embed_dim, out_feats=embed_dim, dropout=dropout))
        self.out_mlp = nn.Sequential(*build_mlp_list(embed_dim, n_mlp=3, dim_mlp=embed_dim, out_feats=embed_dim, dropout=dropout))

        self.outer_cat = OuterConcatLayer()

        return

    def forward(self, query, key, value, need_weights=True):
        """
        query of size tgt_len, bsz, embed_dim
        key of size src_len, bsz, embed_dim
        value of size src_len, bsz, embed_dim
        if need weights is True, returns the attention map as second return value
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert key.size() == value.size()

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        # scaling = float(head_dim) ** -0.5

        # q = self.q_proj(query)
        # k = self.k_proj(key)
        v = self.v_mlp(value)
        # q = q * scaling
        # TODO scaling in the pairwise mlp case?

        # q = q.contiguous().view(tgt_len, bsz*num_heads, head_dim).transpose(0, 1)
        # k = k.contiguous().view(src_len, bsz*num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, head_dim).transpose(0, 1)

        # NOTE replace outer product with outer concat
        # attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = self.outer_cat(query, key)
        # NOTE then do pair-wise mlp
        attn_output_weights = self.attn_mlp(attn_output_weights)  # l_tgt, l_src, bsz, num_heads
        attn_output_weights = attn_output_weights.contiguous().view(tgt_len, src_len, bsz * self.num_heads)  # fuse last dimension TODO check if this actually puts batch and heads in the correct spot
        attn_output_weights = attn_output_weights.transpose(0, 2).transpose(1, 2)  # bsz*num_heads, tgt_len, src_len
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)  # softmax over src_len
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)  # bsz*num_heads, tgt_len, head_dim
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # tgt_len, bsz, embed_dim
        attn_output = self.out_mlp(attn_output)

        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads  # squeezes heads together for easier visualization
        else:
            return attn_output, None


class OuterConcatAttention(nn.Module):
    ''' Attention using outer concatenation

    Instead of the normal multiplication of the Query and the Key,
    this will apply an MLP to each input pair (each Q and K pair).
    The resulting pair-wise attention map will then have the usual
    Value multiplied to return the original dimensions (with a possibly
    changed colour dimension).
    '''
    def __init__(self, in_feats, out_feats, n_mlp=1, dim_mlp=4, dropout=0.0):
        '''
        Inputs:
            - n_mlp (int): Number of MLP layers to apply to QK pairs.
            - dim_mlp (int): Number of nodes per MLP layer.
            - dropout (float): Dropout to apply after MLP layers.
        '''
        super(OuterConcatAttention, self).__init__()
        self.outercat = OuterConcatLayer()
        # Need to add n_mlp number of these
        # The 2*in_feats is because this comes after the outer concat
        self.attn_mlp = nn.Sequential(
            *build_mlp_list(in_feats=2 * in_feats, out_feats=out_feats, n_mlp=n_mlp, dim_mlp=dim_mlp, dropout=dropout)
        )

        # Also build one to apply to the Value
        # This MUST have the same number of output dims as the attn_mlp
        self.value_mlp = nn.Sequential(
            *build_mlp_list(in_feats=in_feats, out_feats=out_feats, n_mlp=n_mlp, dim_mlp=dim_mlp, dropout=dropout)
        )

    def forward(self, x):
        '''
        Shape:
            - Input: (l, N, C)
            - Output (l, N, C_{out})
        '''
        l, N, C = x.size()

        # Produce stacked pairs of leaves
        attn = self.outercat(x)  # l, l, N, 2*C
        attn = attn.reshape(-1, N, 2 * C)  # l^2, N, 2*C

        # Apply our MLP to each pair
        attn = self.attn_mlp(attn)  # l^2, N, C_{out}

        # And reshape to enable multiplication with the Value
        attn = attn.reshape(l, l, N, -1)  # l, l, N, C_{out}

        # Should be a softmax in here

        # Construct the value to multiply
        value = self.value_mlp(x)  # l, N, C_{out}

        # And multiply away
        x = torch.einsum(
            'ijbc,jkbc->ikbc',
            attn,
            value.unsqueeze(1)
        ).squeeze(1)  # l, N, C_{out}

        return x

#!/usr/bin/env python3
'''
Useful functions to construct common layer configurations.
'''
from torch import nn


def build_mlp_list(in_feats, out_feats, n_mlp, dim_mlp, dropout=0.0):
    ''' Construct the ModuleList for MLPs to apply to each node

    The final linear layer will have no activation applied.
    '''
    assert n_mlp > 0, 'Can\'t build an MLP with no layers bruh.'

    mlp = nn.ModuleList()

    # Build list of MLP feature sizes
    feats = [in_feats] + [dim_mlp] * (n_mlp - 1) + [out_feats]

    for i in range(n_mlp - 1):
        mlp.extend([
            nn.Linear(feats[i], feats[i + 1]),
            nn.ELU(),
            # nn.BatchNorm1d(feats[i+1]),
            nn.Dropout(p=dropout),
        ])

    # Final layer will have no activation
    mlp.extend([nn.Linear(feats[-2], feats[-1])])

    return mlp

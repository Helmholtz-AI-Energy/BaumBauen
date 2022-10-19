#!/usr/bin/env python3
'''
A place to put baseline models.
'''

import torch
from torch import nn

from baumbauen.layers import BridgeLayer, build_mlp_list


class BBTransformerBaseline(nn.Module):
    '''
    This model feeds the set of leaves through a normal transformer encoder
    '''
    def __init__(
        self,
        infeatures,
        num_classes,
        nattn=1,
        nhead=4,
        emb_dim=8,
        dim_feedforward=128,
        final_mlp_layers=2,
        dropout=0.3,
        transformer='encoder',
        bridge_method='sum',
        **kwargs,
    ):
        super(BBTransformerBaseline, self).__init__()

        self.mlp0 = nn.Sequential(*build_mlp_list(
            in_feats=infeatures,
            out_feats=emb_dim,
            n_mlp=final_mlp_layers,
            dim_mlp=emb_dim,
            dropout=dropout,
        ))
        # self.embedding = nn.Linear(infeatures, emb_dim)
        self.f0 = nn.LeakyReLU(0.1)

        self.transformer = transformer
        self.bridge_method = bridge_method

        if transformer == 'encoder':
            attns = torch.nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=emb_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation='relu',
                ) for _ in range(nattn)
            ])
            self.attns = nn.Sequential(*attns)

            self.bridge = BridgeLayer(method=bridge_method, format='NHWC')
        else:
            self.trans = nn.Transformer(
                d_model=emb_dim,
                nhead=nhead,
                num_encoder_layers=nattn,
                num_decoder_layers=nattn,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.bridge = BridgeLayer(method=bridge_method, format='LNC')
            if bridge_method == 'concat':
                self.mlp_bridge = nn.Sequential(*build_mlp_list(
                    in_feats=int(2 * emb_dim),
                    out_feats=emb_dim,
                    n_mlp=final_mlp_layers,
                    dim_mlp=emb_dim,
                    dropout=dropout,
                ))
                self.f_bridge = nn.LeakyReLU(0.1)

        # The bridge layer doubles the feature dimension if concat method is used
        if (bridge_method != 'concat' and transformer == 'encoder') or (bridge_method == 'concat' and transformer != 'encoder'):
            final_in_feats = emb_dim
        else:
            final_in_feats = 2 * emb_dim

        self.mlp1 = nn.Sequential(*build_mlp_list(
            in_feats=final_in_feats,
            out_feats=num_classes,
            n_mlp=final_mlp_layers,
            dim_mlp=int(dim_feedforward / 2),
            dropout=dropout,
        ))
        # self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        """
        input of shape (l, b, c),
        output of shape (b, c, l, l)
        """
        x = self.mlp0(x)  # (l, b, d)
        # x = self.embedding(x)  # (l, b, d)
        x = self.f0(x)  # (l, b, d)

        if self.transformer == 'encoder':
            # This needs residuals
            x = self.attns(x)  # (l, b, d)
            # This already symmetrises the output
            x = self.bridge(x)  # (b, l, l, d), if 'concat': (b, l, l, 2d)
        else:
            l, b, d = x.size()
            bridged = self.bridge(x)  # (l*l, b, d), if 'concat': (l*l, b, 2d)
            if self.bridge_method == 'concat':
                bridged = self.mlp_bridge(bridged)  # (l*l, b, d)
                bridged = self.f_bridge(bridged)  # (l*l, b, d)

            x = self.trans(x, bridged)  # In: (l, b, d), (l^2, b, d), Out: (l^2, b, d)

            x = x.reshape(l, l, b, d).permute(2, 0, 1, 3)  # (b, l, l, d)

        # Could apply more feedforward layers here too
        x = self.mlp1(x)  # (b, l, l, c), c=num_classes
        # x = self.fc(x)  # (b, l, l, c), c=num_classes
        # Need in the order for cross entropy loss
        x = x.permute(0, 3, 1, 2)  # (b, c, l, l)

        x = torch.div(x + torch.transpose(x, 2, 3), 2)

        # Don't need to softmax if it's done in the loss
        # TODO: Add softmax as an optional
        return x

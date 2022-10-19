import torch
from torch import nn
from baumbauen.layers import MLP
from baumbauen.utils.data_utils import construct_rel_recvs, construct_rel_sends


class NRIModel(nn.Module):
    ''' NRI model built off the official implementation.

    Contains adaptations to make it work with our use case, plus options for extra layers to give it some more oomph

    Args:
        infeatures (int): Number of input features
        num_classes (int): Number of classes in ouput prediction
        nblocks (int): Number of NRI blocks in the model
        dim_feedforward (int): Width of feedforward layers
        initial_mlp_layers (int): Number of MLP (2 feedforward, 1 batchnorm (optional)) before NRI blocks
        block_additional_mlp_layers (int): Number of additional MLP (2 feedforward, 1 batchnorm (optional)) within NRI blocks, when 0 the total number is one.
        final_mlp_layers (int): Number of MLP (2 feedforward, 1 batchnorm (optional)) after NRI blocks
        dropout (float): Dropout rate
        factor (bool): Whether to use NRI blocks at all (useful for benchmarking)
        tokenize ({int: int}): Dictionary of tokenized features to embed {index_of_feature: num_tokens}
        embedding_dims (int): Number of embedding dimensions to use for tokenized features
        batchnorm (bool): Whether to use batchnorm in MLP layers
    '''
    def __init__(
        self,
        infeatures,
        num_classes,
        nblocks=1,
        dim_feedforward=128,
        initial_mlp_layers=1,
        block_additional_mlp_layers=1,
        final_mlp_layers=1,
        dropout=0.3,
        factor=True,
        tokenize=None,
        embedding_dims=None,
        batchnorm=True,
        symmetrize=True,
        **kwargs,
    ):
        super(NRIModel, self).__init__()

        assert dim_feedforward % 2 == 0, 'dim_feedforward must be an even number'

        self.num_classes = num_classes
        self.factor = factor
        self.tokenize = tokenize
        self.symmetrize = symmetrize
        self.block_additional_mlp_layers = block_additional_mlp_layers
        # self.max_leaves = max_leaves

        # Set up embedding for tokens and adjust input dims
        if self.tokenize is not None:
            assert (embedding_dims is not None) and isinstance(embedding_dims, int), 'embedding_dims must be set to an integer is tokenize is given'

            # Initialise the embedding layers, ignoring pad values
            self.embed = nn.ModuleDict({})
            for idx, n_tokens in self.tokenize.items():
                # NOTE: This assumes a pad value of 0 for the input array x
                self.embed[str(idx)] = nn.Embedding(n_tokens, embedding_dims, padding_idx=0)

            # And update the infeatures to include the embedded feature dims and delete the original, now tokenized feats
            infeatures = infeatures + (len(self.tokenize) * (embedding_dims - 1))
            print(f'Set up embedding for {len(self.tokenize)} inputs')

        # Create first half of inital NRI half-block to go from leaves to edges
        initial_mlp = [MLP(infeatures, dim_feedforward, dim_feedforward, dropout, batchnorm)]
        # Add any additional layers as per request
        initial_mlp.extend([
            MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm) for _ in range(initial_mlp_layers - 1)
        ])
        self.initial_mlp = nn.Sequential(*initial_mlp)

        # MLP to reduce feature dimensions from first Node2Edge before blocks begin
        self.pre_blocks_mlp = MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm)

        if self.factor:
            # MLPs within NRI blocks
            # The blocks have minimum 1 MLP layer, and if specified they add more with a skip connection
            # List of blocks
            self.blocks = nn.ModuleList([
                # List of MLP sequences within each block
                nn.ModuleList([
                    # MLP layers before Edge2Node (start of block)
                    nn.ModuleList([
                        MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm),
                        nn.Sequential(*[MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm) for _ in range(block_additional_mlp_layers)]),
                        # This is what would be needed for a concat instead of addition of the skip connection
                        # MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm) if (block_additional_mlp_layers > 0) else None,
                    ]),
                    # MLP layers between Edge2Node and Node2Edge (middle of block)
                    nn.ModuleList([
                        MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm),
                        nn.Sequential(*[MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm) for _ in range(block_additional_mlp_layers)]),
                        # This is what would be needed for a concat instead of addition of the skip connection
                        # MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm) if (block_additional_mlp_layers > 0) else None,
                    ]),
                    # MLP layer after Node2Edge (end of block)
                    # This is just to reduce feature dim after skip connection was concatenated
                    MLP(dim_feedforward * 3, dim_feedforward, dim_feedforward, dropout, batchnorm),
                ]) for _ in range(nblocks)
            ])
            print("Using factor graph MLP encoder.")
        else:
            self.mlp3 = MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm)
            self.mlp4 = MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm)
            print("Using MLP encoder.")

        # Final linear layers as requested
        # self.final_mlp = nn.Sequential(*[MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm) for _ in range(final_mlp_layers)])
        final_mlp = [MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm)]
        # Add any additional layers as per request
        final_mlp.extend([
            MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm) for _ in range(final_mlp_layers - 1)
        ])
        self.final_mlp = nn.Sequential(*final_mlp)

        self.fc_out = nn.Linear(dim_feedforward, self.num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec):
        '''
        Input: (b, l*l, d), (b, l*l, l)
        Output: (b, l, d)
        '''
        # TODO assumes that batched matrix product just works
        # TODO these do not have to be members
        incoming = torch.matmul(rel_rec.permute(0, 2, 1), x)  # (b, l, d)
        denom = rel_rec.sum(1)[:, 1]
        return incoming / denom.reshape(-1, 1, 1)  # (b, l, d)
        # return incoming / incoming.size(1)  # (b, l, d)

    def node2edge(self, x, rel_rec, rel_send):
        '''
        Input: (b, l, d), (b, l*(l-1), l), (b, l*(l-1), l)
        Output: (b, l*l(l-1), 2d)
        '''
        # TODO assumes that batched matrix product just works
        receivers = torch.matmul(rel_rec, x)  # (b, l*l, d)
        senders = torch.matmul(rel_send, x)  # (b, l*l, d)
        edges = torch.cat([senders, receivers], dim=2)  # (b, l*l, 2d)

        return edges

    def forward(self, inputs):
        '''
        Input: (l, b, d)
        Output: (b, c, l, l)
        '''

        if isinstance(inputs, (list, tuple)):
            inputs, rel_rec, rel_send = inputs
        else:
            rel_rec = None
            rel_send = None

        n_leaves, batch, feats = inputs.size()
        device = inputs.device

        # NOTE create rel matrices on the fly if not given as input
        if rel_rec is None:
            # rel_rec = torch.eye(
            #     n_leaves,
            #     device=device
            # ).repeat_interleave(n_leaves-1, dim=1).T  # (l*(l-1), l)
            # rel_rec = rel_rec.unsqueeze(0).expand(inputs.size(1), -1, -1)
            rel_rec = construct_rel_recvs([inputs.size(0)], device=device)

        if rel_send is None:
            # rel_send = torch.eye(n_leaves, device=device).repeat(n_leaves, 1)
            # rel_send[torch.arange(0, n_leaves*n_leaves, n_leaves + 1)] = 0
            # rel_send = rel_send[rel_send.sum(dim=1) > 0]  # (l*(l-1), l)
            # rel_send = rel_send.unsqueeze(0).expand(inputs.size(1), -1, -1)
            rel_send = construct_rel_sends([inputs.size(0)], device=device)

        # Input shape: [batch, num_atoms, num_timesteps, num_dims]
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        # x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        # Need to match expected shape
        # TODO should batch_first be a dataset parameter?
        x = inputs.permute(1, 0, 2)  # (b, l, d)

        # Create embeddings and merge back into x
        # TODO: Move mask creation to init, optimise this loop
        if self.tokenize is not None:
            emb_x = []
            # We'll use this to drop tokenized features from x
            mask = torch.ones(feats, dtype=torch.bool, device=device)
            for idx, emb in self.embed.items():
                # Note we need to convert tokens to type long here for embedding layer
                emb_x.append(emb(x[..., int(idx)].long()))  # List of (b, l, emb_dim)
                mask[int(idx)] = False

            # Now merge the embedding outputs with x (mask has deleted the old tokenized feats)
            x = torch.cat([x[..., mask], *emb_x], dim=-1)  # (b, l, d + embeddings)
            del emb_x

        # Initial set of linear layers
        x = self.initial_mlp(x)  # Series of 2-layer ELU net per node  (b, l, d) optionally includes embeddings

        x = self.node2edge(x, rel_rec, rel_send)  # (b, l*l, 2d)

        # All things related to NRI blocks are in here
        if self.factor:
            x = self.pre_blocks_mlp(x)  # (b, l*l, d)
            # Skip connection to jump over all NRI blocks
            x_global_skip = x

            for block in self.blocks:
                x_skip = x  # (b, l*l, d)

                # First MLP sequence
                x = block[0][0](x)  # (b, l*l, d)
                if self.block_additional_mlp_layers > 0:
                    x_first_skip = x  # (b, l*l, d)
                    x = block[0][1](x)  # (b, l*l, d)
                    x = x + x_first_skip  # (b, l*l, d)
                    del x_first_skip

                # Create nodes from edges
                x = self.edge2node(x, rel_rec)  # (b, l, d)

                # Second MLP sequence
                x = block[1][0](x)  # (b, l, d)
                if self.block_additional_mlp_layers > 0:
                    x_second_skip = x  # (b, l*l, d)
                    x = block[1][1](x)  # (b, l*l, d)
                    x = x + x_second_skip  # (b, l*l, d)
                    del x_second_skip

                # Create edges from nodes
                x = self.node2edge(x, rel_rec, rel_send)  # (b, l*l, 2d)

                # Final MLP in block to reduce dimensions again
                x = torch.cat((x, x_skip), dim=2)  # Skip connection  # (b, l*l, 3d)
                x = block[2](x)  # (b, l*l, d)
                del x_skip

            # Global skip connection
            x = torch.cat((x, x_global_skip), dim=2)  # Skip connection  # (b, l*(l-1), 2d)

            # Cleanup
            del rel_rec, rel_send

        # else:
        #     x = self.mlp3(x)  # (b, l*(l-1), d)
        #     x = torch.cat((x, x_skip), dim=2)  # Skip connection  # (b, l*(l-1), 2d)
        #     x = self.mlp4(x)  # (b, l*(l-1), d)

        # Final set of linear layers
        x = self.final_mlp(x)  # Series of 2-layer ELU net per node (b, l, d)

        # Output what will be used for LCA
        x = self.fc_out(x)  # (b, l*l, c)
        out = x.reshape(batch, n_leaves, n_leaves, self.num_classes)

        # Change to LCA shape
        # x is currently the flattened rows of the predicted LCA but without the diagonal
        # We need to create an empty LCA then populate the off-diagonals with their corresponding values
        # out = torch.zeros((batch, n_leaves, n_leaves, self.num_classes), device=device)  # (b, l, l, c)
        # ind_upper = torch.triu_indices(n_leaves, n_leaves, offset=1)
        # ind_lower = torch.tril_indices(n_leaves, n_leaves, offset=-1)

        # Assign the values to their corresponding position in the LCA
        # The right side is just some quick mafs to get the correct edge predictions from the flattened x array
        # out[:, ind_upper[0], ind_upper[1], :] = x[:, (ind_upper[0] * (n_leaves - 1)) + ind_upper[1] - 1, :]
        # out[:, ind_lower[0], ind_lower[1], :] = x[:, (ind_lower[0] * (n_leaves - 1)) + ind_lower[1], :]

        # Need in the order for cross entropy loss
        out = out.permute(0, 3, 1, 2)  # (b, c, l, l)

        # Symmetrize
        if self.symmetrize:
            out = torch.div(out + torch.transpose(out, 2, 3), 2)  # (b, c, l, l)

        return out

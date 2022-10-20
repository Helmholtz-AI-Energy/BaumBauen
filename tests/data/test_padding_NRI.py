import torch
import baumbauen as bb
import unittest
from baumbauen.models import NRIModel
from baumbauen.utils.data_utils import construct_rel_recvs, construct_rel_sends


class TestNRIPadding(unittest.TestCase):
    ''' Tests to check that padding has no effect on NRI output/components '''
    def test_same_output_ex1(self):

        # unpadded feature example.target, rel_rec and rel_send
        leaves_1 = 5
        feature_1 = torch.randn((leaves_1, 4))
        # Create a trivial target
        target_1 = (-1 * torch.ones(leaves_1)) - torch.eye(leaves_1)

        leaves_2 = 10
        feature_2 = torch.randn((leaves_2, 4))
        # Create a trivial target
        target_2 = (-1 * torch.ones(leaves_2)) - torch.eye(leaves_2)

        # dataset = bb.datasets.TreeSet([feature_1, feature_2], [target_1, target_2])
        dataset = [(feature_1, target_1), (feature_2, target_2)]

        # Batch size 1 means no padding
        dataloader_unpadded = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            drop_last=False,
            shuffle=False,
            collate_fn=bb.utils.rel_pad_collate_fn,
        )
        # Padding due to more leaves in feature_2
        dataloader_padded = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            drop_last=False,
            shuffle=False,
            collate_fn=bb.utils.rel_pad_collate_fn,
        )

        model = bb.models.NRIModel(
            infeatures=4,
            num_classes=2,
            nblocks=2,
            dim_feedforward=8,
            bnorm=True,  # currently ignored
        )

        model.eval()

        # Perform inference
        with torch.no_grad():
            # These both return shapes (N, C, L, L)
            pred_unpadded = model(next(iter(dataloader_unpadded))[0])
            pred_padded = model(next(iter(dataloader_padded))[0])

        # Get the first (padded) sample from the batch
        pred_unpadded = pred_unpadded[0]  # (C, L, L)
        # for the padded one, only extract the non-padding components
        pred_padded = pred_padded[0, :, :5, :5]  # (C, L, L)

        # torch.equal isnt working here because of floats,
        self.assertTrue(torch.allclose(pred_unpadded, pred_padded, atol=1e-05))

    def test_same_output_ex2(self):
        ''' Same test as above but using pad components '''

        L, B, D = (2, 1, 4)
        depth = 2
        padded_L = 3

        leaves = torch.arange(B * L * D, dtype=torch.float).reshape((L, B, D))

        padded_leaves = torch.zeros(padded_L, B, D)
        padded_leaves[:L, :, :] = leaves[:, :, :]

        model = NRIModel(4, depth, dim_feedforward=8)
        model.eval()

        output = model(leaves)

        padded_rel_recvs = construct_rel_recvs([L, padded_L])[0:1]
        padded_rel_sends = construct_rel_sends([L, padded_L])[0:1]

        padded_output = model((padded_leaves, padded_rel_recvs, padded_rel_sends))

        self.assertTrue(torch.isclose(output, padded_output[:, :, :L, :L]).all())

    def test_node2edge_padding(self):
        depth = 2
        model = NRIModel(4, depth)
        for L in [2, 3, 4, 5]:
            for padded_L in range(L, 7):
                B, D = (1, 4)

                nodes = torch.arange((B * L * D), dtype=torch.float).reshape((B, L, D))
                rel_recvs = construct_rel_recvs([L])
                rel_sends = construct_rel_sends([L])

                edges = model.node2edge(nodes, rel_recvs, rel_sends)

                padded_nodes = torch.zeros(B, padded_L, D)
                padded_nodes[0, :L] = nodes[0, :]
                padded_rel_recvs = construct_rel_recvs([L, padded_L])[0:1]
                padded_rel_sends = construct_rel_sends([L, padded_L])[0:1]

                padded_edges = model.node2edge(padded_nodes, padded_rel_recvs, padded_rel_sends)
                self.assertTrue((edges[edges.sum(dim=2) > 0] == padded_edges[padded_edges.sum(dim=2) > 0]).all())
                # self.assertTrue((edges == padded_edges[]).all())


    def test_edge2node_padding(self):
        depth = 2
        model = NRIModel(4, depth)
        for L in [2, 3, 4, 5]:
            for padded_L in range(L, 7):
                B, D = (1, 4)

                edges = torch.arange((B * L * L * D), dtype=torch.float).reshape((B, L * L, D))
                rel_recvs = construct_rel_recvs([L])
                nodes = model.edge2node(edges, rel_recvs)

                padded_edges = torch.zeros(B, padded_L * padded_L , D)
                for i in range(L * L):
                    padded_edges[:, padded_L * (i // L) + i % L] = edges[:, i]

                padded_rel_recvs = construct_rel_recvs([L, padded_L])[0:1]

                padded_nodes = model.edge2node(padded_edges, padded_rel_recvs)

                self.assertTrue((nodes == padded_nodes[0, :L]).all())

    def test_node2edge2node_padding(self):
        depth = 2
        model = NRIModel(4, depth)
        for L in [2, 3, 4, 5]:
            for padded_L in range(L, 7):
                L, B, D = (2, 1, 4)
                padded_L = 3

                nodes = torch.arange(B * L * D, dtype=torch.float).reshape((B, L, D))
                rel_recvs = construct_rel_recvs([L])
                rel_sends = construct_rel_sends([L])

                edges = model.node2edge(nodes, rel_recvs, rel_sends)

                padded_nodes = torch.zeros(B, padded_L, D)
                padded_nodes[0, :L] = nodes[0, :]
                padded_rel_recvs = construct_rel_recvs([L, padded_L])[0:1]
                padded_rel_sends = construct_rel_sends([L, padded_L])[0:1]

                padded_edges = model.node2edge(padded_nodes, padded_rel_recvs, padded_rel_sends)

                nodes_out = model.edge2node(edges, rel_recvs)
                padded_nodes_out = model.edge2node(padded_edges, padded_rel_recvs)

                self.assertTrue((nodes_out == padded_nodes_out[0, :L]).all())


if __name__ == '__main__':
    unittest.main()

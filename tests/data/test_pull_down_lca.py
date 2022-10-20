import torch as t
import numpy as np

import unittest

from baumbauen.utils import pull_down_LCA


class PullDownLCATest(unittest.TestCase):
    def test_torch_1(self):

        lca = t.tensor([
            [0, 1],
            [4, 5],
        ])

        lca = pull_down_LCA(lca)

        comparison_lca = t.tensor([
            [0, 1],
            [2, 3],
        ])

        self.assertTrue((lca == comparison_lca).all())

    def test_numpy_1(self):

        lca = np.array([
            [0, 1],
            [4, 5],
        ])

        lca = pull_down_LCA(lca)

        comparison_lca = np.array([
            [0, 1],
            [2, 3],
        ])

        self.assertTrue((lca == comparison_lca).all())

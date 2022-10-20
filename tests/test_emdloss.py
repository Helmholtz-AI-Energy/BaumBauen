import torch as t
import unittest

from baumbauen.losses import EMDLoss


class EMDLossTest(unittest.TestCase):
    def test_valid_example1(self):
        ''' Trivial case, perfect classification '''

        example_input_1 = t.tensor([[[1.]]], dtype=t.float32)
        example_target_1 = t.tensor([[0]], dtype=t.long)

        emd_loss = EMDLoss()
        comparison = emd_loss(example_input_1, example_target_1)

        self.assertTrue((comparison == 0.).all())

    def test_valid_example2(self):
        ''' Normal classification, L1 norm, no normalisation'''

        example_input_1 = t.tensor([[[0.5], [0.5]]], dtype=t.float32)
        example_target_1 = t.tensor([[0]], dtype=t.long)

        emd_loss = EMDLoss(l=1, normalise=False)
        comparison = emd_loss(example_input_1, example_target_1)

        self.assertTrue((comparison == 0.5).all())

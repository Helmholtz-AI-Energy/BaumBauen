import torch as t
import unittest

from baumbauen.metrics import Efficiency


class EfficiencyMetricTest(unittest.TestCase):
    def test_single_valid_1(self):

        example_1 = t.tensor([
            # b   c   d
            [-1,  1,  1],  # b
            [ 1, -1,  1],  # c
            [ 1,  1, -1]   # d
        ])
        example_1 = t.unsqueeze(example_1, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1].permute(0, 3, 1, 2)

        eff = Efficiency()
        eff.reset()
        eff.update((example_1_preds, example_1))

        self.assertAlmostEqual(eff.compute(), 1)

    def test_single_invalid_1(self):

        example_1 = t.tensor([
            # b   c   d
            [-1,  2,  1],  # b
            [ 2, -1,  1],  # c
            [ 1,  1, -1]   # d
        ])
        example_1 = t.unsqueeze(example_1, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1].permute(0, 3, 1, 2)

        eff = Efficiency()
        eff.reset()
        eff.update((example_1_preds, example_1))

        self.assertAlmostEqual(eff.compute(), 0)

    def test_single_invalid_2(self):
        ''' Empty LCAs are technically a valid tree if we allow disconnected leaves but are useless so we don't count them '''

        example_1 = t.tensor([
            #b  c  d
            [-1,  0,  0],  # b
            [ 0, -1,  0],  # c
            [ 0,  0, -1]   # d
        ])
        example_1 = t.unsqueeze(example_1, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1].permute(0, 3, 1, 2)

        eff = Efficiency()
        eff.reset()
        eff.update((example_1_preds, example_1))

        self.assertAlmostEqual(eff.compute(), 0)

    def test_single_valid_padded_1(self):

        example_1 = t.tensor([
            # b   c   d   p
            [-1,  1,  1, -1],  # b
            [ 1, -1,  1, -1],  # c
            [ 1,  1, -1, -1],  # d
            [-1, -1, -1, -1],  # p
        ])
        example_1 = t.unsqueeze(example_1, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1].permute(0, 3, 1, 2)

        eff = Efficiency(ignore_index=-1)
        eff.reset()
        eff.update((example_1_preds, example_1))

        self.assertAlmostEqual(eff.compute(), 1)

    def test_single_valid_padded_2(self):

        example_2 = t.tensor([
            # b   c   d   p
            [-1,  1,  1, -1],  # b
            [ 1, -1,  1, -1],  # c
            [ 1,  1, -1, -1],  # d
            [-1, -1, -1, -1],  # p
        ])
        example_2 = t.unsqueeze(example_2, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_2.unique())
        example_2_preds = t.eye(classes)[example_2].permute(0, 3, 1, 2)

        eff = Efficiency(ignore_index=-1.)
        eff.reset()
        eff.update((example_2_preds, example_2))

        self.assertAlmostEqual(eff.compute(), 1)

    def test_single_valid_padded_3(self):
        ''' This is testing multiple padding values '''

        example_3 = t.tensor([
            # b   c   d   p
            [-1,  1,  1, -1],  # b
            [ 1, -1,  1, -2],  # c
            [ 1,  1, -1, -2],  # d
            [-1, -2, -2, -1],  # p
        ])
        example_3 = t.unsqueeze(example_3, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_3.unique())
        example_3_preds = t.eye(classes)[example_3].permute(0, 3, 1, 2)

        eff = Efficiency(ignore_index=[-1., -2.])
        eff.reset()
        eff.update((example_3_preds, example_3))

        self.assertAlmostEqual(eff.compute(), 1)

    def test_single_valid_disconnected_leaves_1(self):
        ''' Testing an LCA that's valid if we ignore disconnected leaves '''

        example_1 = t.tensor([
            # b   c   d   m
            [-1,  1,  1,  0],  # b
            [ 1, -1,  1,  0],  # c
            [ 1,  1, -1,  0],  # d
            [ 0,  0,  0, -1],  # m
        ])
        example_1 = t.unsqueeze(example_1, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1].permute(0, 3, 1, 2)

        eff = Efficiency(ignore_index=-1., ignore_disconnected_leaves=True)
        eff.reset()
        eff.update((example_1_preds, example_1))

        self.assertAlmostEqual(eff.compute(), 1)

    def test_single_invalid_disconnected_leaves_1(self):
        ''' Testing an LCA that's invalid if we don't ignore disconnected leaves '''

        example_1 = t.tensor([
            # b   c   d   m
            [-1,  1,  1,  0],  # b
            [ 1, -1,  1,  0],  # c
            [ 1,  1, -1,  0],  # d
            [ 0,  0,  0, -1],  # m
        ])
        example_1 = t.unsqueeze(example_1, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1].permute(0, 3, 1, 2)

        eff = Efficiency(ignore_index=-1., ignore_disconnected_leaves=False)
        eff.reset()
        eff.update((example_1_preds, example_1))

        self.assertAlmostEqual(eff.compute(), 0)

    def test_single_invalid_disconnected_leaves_2(self):
        ''' Testing an LCA that's invalid even if we ignore disconnected leaves '''

        example_2 = t.tensor([
            # b   c   d   m
            [-1,  1,  1,  1],  # b
            [ 1, -1,  1,  0],  # c
            [ 1,  1, -1,  0],  # d
            [ 1,  0,  0, -1],  # m
        ])
        example_2 = t.unsqueeze(example_2, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_2.unique())
        example_2_preds = t.eye(classes)[example_2].permute(0, 3, 1, 2)

        eff = Efficiency(ignore_index=-1., ignore_disconnected_leaves=True)
        eff.reset()
        eff.update((example_2_preds, example_2))

        self.assertAlmostEqual(eff.compute(), 0)

    def test_single_invalid_disconnected_leaves_3(self):
        ''' Testing a trivial LCA that we regard as invalid even if we ignore disconnected leaves '''

        example_2 = t.tensor([
            # b   c   d   m
            [-1,  0,  0,  0],  # b
            [ 0, -1,  0,  0],  # c
            [ 0,  0, -1,  0],  # d
            [ 0,  0,  0, -1],  # m
        ])
        example_2 = t.unsqueeze(example_2, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_2.unique())
        example_2_preds = t.eye(classes)[example_2].permute(0, 3, 1, 2)

        eff = Efficiency(ignore_index=-1., ignore_disconnected_leaves=True)
        eff.reset()
        eff.update((example_2_preds, example_2))

        self.assertAlmostEqual(eff.compute(), 0)

    def test_multiple_1(self):

        example_1 = t.tensor([[
            # b  c  d
            [-1,  1,  1],  # b
            [ 1, -1,  1],  # c
            [ 1,  1, -1],  # d
        ],[
            # b   c  d
            [-1,  1, 0],  # b
            [ 1, -1, 1],  # c
            [-1,  1, 0],  # d
        ]])

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1].permute(0, 3, 1, 2)

        eff = Efficiency()
        eff.reset()
        eff.update((example_1_preds, example_1))

        self.assertAlmostEqual(eff.compute(), 0.5)


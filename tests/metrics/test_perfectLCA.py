import torch as t
import unittest

from baumbauen.metrics import PerfectLCA


class PerfectLCAMetricTest(unittest.TestCase):
    def test_single_perfect_1(self):

        example_1 = t.tensor([
            #b  c  d
            [0, 1, 1],  # b
            [1, 0, 1],  # c
            [1, 1, 0]   # d
        ])
        example_1 = t.unsqueeze(example_1, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1].permute(0, 3, 1, 2)

        perf = PerfectLCA()
        perf.reset()
        perf.update((example_1_preds, example_1))

        self.assertAlmostEqual(perf.compute(), 1)

    def test_single_imperfect_1(self):

        example_1 = t.tensor([
            #b  c  d
            [0, 2, 1],  # b
            [2, 0, 1],  # c
            [1, 1, 0]   # d
        ])
        example_1 = t.unsqueeze(example_1, 0)

        example_1_preds = t.tensor([
            #b  c  d
            [0, 1, 1],  # b
            [1, 0, 1],  # c
            [1, 1, 0]   # d
        ])
        example_1_preds = t.unsqueeze(example_1_preds, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1_preds].permute(0, 3, 1, 2)

        perf = PerfectLCA()
        perf.reset()
        perf.update((example_1_preds, example_1))

        self.assertAlmostEqual(perf.compute(), 0)

    def test_single_perfect_padded_1(self):

        example_1 = t.tensor([
            #b  c  d  p
            [0, 1, 1, 0],  # b
            [1, 0, 1, 0],  # c
            [1, 1, 0, 0],  # d
            [0, 0, 0, 0],  # p
        ])
        example_1 = t.unsqueeze(example_1, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1].permute(0, 3, 1, 2)

        perf = PerfectLCA(ignore_index=0)
        perf.reset()
        perf.update((example_1_preds, example_1))

        self.assertAlmostEqual(perf.compute(), 1)

    def test_single_imperfect_padded_1(self):

        example_1 = t.tensor([
            #b  c  d  p
            [0, 2, 1, 0],  # b
            [2, 0, 1, 0],  # c
            [1, 1, 0, 0],  # d
            [0, 0, 0, 0],  # p
        ])
        example_1 = t.unsqueeze(example_1, 0)

        example_1_preds = t.tensor([
            #b  c  d  p
            [0, 1, 1, 0],  # b
            [1, 0, 1, 0],  # c
            [1, 1, 0, 0],  # d
            [0, 0, 0, 0],  # p
        ])
        example_1_preds = t.unsqueeze(example_1_preds, 0)

        # Convert to a prediction (doesn't matter if correct)
        classes = len(example_1.unique())
        example_1_preds = t.eye(classes)[example_1_preds].permute(0, 3, 1, 2)

        perf = PerfectLCA(ignore_index=0)
        perf.reset()
        perf.update((example_1_preds, example_1))

        self.assertAlmostEqual(perf.compute(), 0)

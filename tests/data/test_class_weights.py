import torch as t
import unittest

from baumbauen.utils import calculate_class_weights
import baumbauen as bb


class ClassWeightsTest(unittest.TestCase):
    def test_trivial_case_1(self):
        '''
        Test the trivial case of a single class
        '''

        num_classes = 1
        x_1 = t.arange(2 * 4).reshape((2,4))
        y_1 = t.tensor([
            # a   b
            [-1,  0],  # a
            [ 0, -1],  # b
        ])

        dataloader = t.utils.data.DataLoader([(x_1, y_1)])

        weights = calculate_class_weights(dataloader, num_classes, num_batches=1)

        comparison_weights = t.tensor([1.])

        self.assertTrue((weights == comparison_weights).all())

    def test_trivial_case_2(self):
        '''
        Test the trivial case of two classes, but one not present in the samples
        '''

        num_classes = 2
        x_1 = t.arange(2 * 4).reshape((2,4))
        y_1 = t.tensor([
            # a   b
            [-1,  1],  # a
            [ 1, -1],  # b
        ])

        dataloader = t.utils.data.DataLoader([(x_1, y_1)])

        weights = calculate_class_weights(dataloader, num_classes, num_batches=1)
        print(weights)

        comparison_weights = t.tensor([1., 1.])

        self.assertTrue((weights == comparison_weights).all())

    def test_trivial_case_3(self):
        '''
        Test the trivial case of two classes
        '''

        num_classes = 2
        x_1 = t.arange(3 * 4).reshape((3,4))
        y_1 = t.tensor([
            # a   b   c
            [-1,  1,  0],  # a
            [ 1, -1,  0],  # b
            [ 0,  0, -1],  # b
        ])

        dataloader = t.utils.data.DataLoader([(x_1, y_1)])

        weights = calculate_class_weights(dataloader, num_classes, num_batches=1)

        comparison_weights = t.tensor([(1. / 3.), (2. / 3.)])

        self.assertTrue((weights == comparison_weights).all())

    def test_trivial_case_4(self):
        '''
        Test the trivial case of two classes, spread across two samples
        '''

        num_classes = 2
        x_1 = t.arange(2 * 4).reshape((2,4))
        y_1 = t.tensor([
            # a   b
            [-1,  0],  # a
            [ 0, -1],  # b
        ])
        y_2 = t.tensor([
            # a   b
            [-1,  1],  # a
            [ 1, -1],  # b
        ])

        dataloader = t.utils.data.DataLoader([(x_1, y_1), (x_1, y_2)])

        weights = calculate_class_weights(dataloader, num_classes, num_batches=2)

        comparison_weights = t.tensor([0.5, 0.5])

        self.assertTrue((weights == comparison_weights).all())

    def test_bb_dataloader(self):
        '''
        Test using baumbauen dataloader with padding
        '''

        num_classes = 3
        x_1 = t.arange(2 * 4).reshape((2,4))
        y_1 = t.tensor([
            # a   b
            [-1,  1],  # a
            [ 1, -1],  # b
        ])
        x_2 = t.arange(3 * 4).reshape((3,4))
        y_2 = t.tensor([
            # a   b   c
            [-1,  1,  2],  # a
            [ 1, -1,  2],  # b
            [ 2,  2, -1],  # b
        ])
        dataset = [(x_1, y_1), (x_2, y_2)]

        # Padding due to more leaves in feature_2
        dataloader = t.utils.data.DataLoader(
            dataset,
            batch_size=2,
            drop_last=False,
            shuffle=False,
            collate_fn=bb.utils.rel_pad_collate_fn,
        )

        weights = calculate_class_weights(dataloader, num_classes, num_batches=1)

        comparison_weights = t.tensor([1., 0.5, 0.5])

        self.assertTrue((weights == comparison_weights).all())

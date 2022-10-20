import torch as t
import unittest

from baumbauen.utils import ordinalise_labels


class OrdinaliseLabelsTest(unittest.TestCase):
    def test_valid_example1(self):
        '''
            3 classes
        '''

        n_classes = 3
        example_1 = t.tensor([
            #a, b, c
            [0, 1, 2]
        ], dtype=t.float32)

        ordinal_matrix = ordinalise_labels(example_1, n_classes)

        comparison = t.tensor([[
            # a       b       c 
            [0.6652, 0.2119, 0.0900],  # class 1
            [0.2447, 0.5761, 0.2447],  # class 2
            [0.0900, 0.2119, 0.6652],  # class 3
        ]])

        # Need a tolerance because I didn't encode all the float precision
        self.assertTrue(comparison.allclose(ordinal_matrix, atol=1e-04))

    # def test_valid_example_2(self):
    #     '''
    #                +---+
    #                | i |
    #                ++-++
    #                 | |
    #          +------+ +-----+
    #          |              |
    #          |            +-+-+
    #          |            | h |
    #          |            ++-++
    #          |             | |
    #          |        +----+ +-+
    #          |        |        |
    #        +-+-+      |      +-+-+
    #        | g |      |      | f |
    #        ++-++      |      ++-++
    #         | |       |       | |
    #       +-+ +-+     |     +-+ +-+
    #       |     |     |     |     |
    #     +-+-+ +-+-+ +-+-+ +-+-+ +-+-+
    #     | a | | b | | c | | d | | e |
    #     +---+ +---+ +---+ +---+ +---+
    #     '''

    #     example_2 = t.tensor([
    #         #a  b  c  d  e  f  g  h  i
    #         [1, 0, 0, 0, 0, 0, 1, 0, 0],  # a
    #         [0, 1, 0, 0, 0, 0, 1, 0, 0],  # b
    #         [0, 0, 1, 0, 0, 0, 0, 1, 0],  # c
    #         [0, 0, 0, 1, 0, 1, 0, 0, 0],  # d
    #         [0, 0, 0, 0, 1, 1, 0, 0, 0],  # e
    #         [0, 0, 0, 1, 1, 1, 0, 1, 0],  # f
    #         [1, 1, 0, 0, 0, 0, 1, 0, 1],  # g
    #         [0, 0, 1, 0, 0, 1, 0, 1, 1],  # h
    #         [0, 0, 0, 0, 0, 0, 1, 1, 1]   # i
    #     ])
    #     lca_matrix = adjacency2lca(example_2)

    #     comparison = t.tensor([
    #         # a  b  c  d  e
    #         [0, 1, 2, 3, 3],  # a
    #         [1, 0, 2, 3, 3],  # b
    #         [2, 2, 0, 2, 2],  # c
    #         [3, 3, 2, 0, 1],  # d
    #         [3, 3, 2, 1, 0]   # e
    #     ])
    #     self.assertTrue((lca_matrix == comparison).all())

    # def test_valid_example_3(self):
    #     '''
    #           +---+
    #           | d |
    #           +-+-+
    #           | | |
    #       +---+ | +---+
    #       |     |     |
    #     +-+-+ +-+-+ +-+-+
    #     | a | | b | | c |
    #     +---+ +---+ +---+
    #     '''
    #     example_3 = t.tensor([
    #         # a  b  c  d
    #         [1, 0, 0, 1],  # a
    #         [0, 1, 0, 1],  # b
    #         [0, 0, 1, 1],  # c
    #         [1, 1, 1, 1]   # d
    #     ])
    #     lca_matrix = adjacency2lca(example_3)

    #     comparison = t.tensor([
    #         #a  b  c
    #         [0, 1, 1],  # a
    #         [1, 0, 1],  # b
    #         [1, 1, 0]   # c
    #     ])
    #     self.assertTrue((lca_matrix == comparison).all())

    # def test_illegal_empty_example(self):
    #     '''
    #     +---+ +---+ +---+
    #     | a | | b | | c |
    #     +---+ +---+ +---+
    #     '''
    #     zero_example = t.tensor([
    #         #a  b  c
    #         [0, 0, 0],  # a
    #         [0, 0, 0],  # b
    #         [0, 0, 0]   # c
    #     ])

    #     with self.assertRaises(InvalidAdjacencyMatrix):
    #         adjacency2lca(zero_example)

    # def test_illegal_floating_leaf(self):
    #     '''
    #              +---+
    #              | d |
    #              ++-++
    #               | |
    #             +-+ +-+
    #             |     |
    #     +---+ +-+-+ +-+-+
    #     | a | | b | | c |
    #     +---+ +---+ +---+
    #     '''
    #     floating_leaf_example = t.tensor([
    #         #a  b  c
    #         [0, 0, 0],  # a
    #         [0, 0, 1],  # b
    #         [0, 1, 0]   # c
    #     ])

    #     with self.assertRaises(InvalidAdjacencyMatrix):
    #         adjacency2lca(floating_leaf_example)

    # def test_illegal_asymmetric(self):
    #     '''
    #     +---+ +---+ +---+
    #     | a | | b | | c |
    #     +---+ +---+ +---+
    #     '''
    #     asymmetric_example = t.tensor([
    #         #a  b  c
    #         [0, 0, 0],  # a
    #         [1, 0, 0],  # b
    #         [0, 0, 0]   # c
    #     ])

    #     with self.assertRaises(InvalidAdjacencyMatrix):
    #         adjacency2lca(asymmetric_example)

    # def test_illegal_connections(self):
    #     illegal_connections_example = t.tensor([
    #         #a  b  c
    #         [0, 2, 1],  # a
    #         [2, 0, 1],  # b
    #         [1, 1, 0]   # c
    #     ])

    #     with self.assertRaises(InvalidAdjacencyMatrix):
    #         adjacency2lca(illegal_connections_example)


if __name__ == '__main__':
    unittest.main()

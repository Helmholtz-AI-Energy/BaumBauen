import torch as t
import unittest

from baumbauen.utils import lca2adjacency
from baumbauen.utils.lca2adjacency import InvalidLCAMatrix


class LCAReconstructionTest(unittest.TestCase):
    def test_valid_shuffled_bfs_example_1(self):
        """
                       +---+
                       | a |
                       ++-++
                        | |
                +-------+ +------+
                |                |
              +-+-+            +-+-+
              | b |            | c |
              +++++            ++-++
               |||              | |
          +----+|+----+     +---+ +---+
          |     |     |     |         |
        +-+-+ +-+-+ +-+-+ +-+-+     +-+-+
        | d | | e | | f | | g |     | h |
        +---+ +---+ +---+ +---+     +---+
        """
        example_1 = t.tensor([
            #d  g  f  h  e
            [0, 2, 1, 2, 1], # d
            [2, 0, 2, 1, 2], # g
            [1, 2, 0, 2, 1], # f
            [2, 1, 2, 0, 2], # h
            [1, 2, 1, 2, 0], # e
        ])
        adjacency = lca2adjacency(example_1, format='bfs')

        comparison = t.tensor([
            #a  b  c  d  e  f  g  h
            [0, 1, 1, 0, 0, 0, 0, 0],  # a
            [1, 0, 0, 1, 1, 1, 0, 0],  # b
            [1, 0, 0, 0, 0, 0, 1, 1],  # c
            [0, 1, 0, 0, 0, 0, 0, 0],  # d
            [0, 1, 0, 0, 0, 0, 0, 0],  # e
            [0, 1, 0, 0, 0, 0, 0, 0],  # f
            [0, 0, 1, 0, 0, 0, 0, 0],  # g
            [0, 0, 1, 0, 0, 0, 0, 0]   # h
        ])
        self.assertTrue((adjacency == comparison).all())

    def test_valid_shuffled_bfs_example_2(self):
        """ Testing with disjointed levels

                       +---+
                       | a |
                       ++-++
                        | |
                +-------+ +------+
                |                |
              +-+-+            +-+-+
              | b |            | c |
              +++++            ++-++
               |||              | |
          +----+|+----+     +---+ +---+
          |     |     |     |         |
        +-+-+ +-+-+ +-+-+ +-+-+     +-+-+
        | d | | e | | f | | g |     | h |
        +---+ +---+ +---+ +---+     +---+
        """
        example_2 = t.tensor([
            #d  g  f  h  e
            [0, 5, 2, 5, 2], # d
            [5, 0, 5, 2, 5], # g
            [2, 5, 0, 5, 2], # f
            [5, 2, 5, 0, 5], # h
            [2, 5, 2, 5, 0], # e
        ])
        adjacency = lca2adjacency(example_2, format='bfs')

        comparison = t.tensor([
            #a  b  c  d  e  f  g  h
            [0, 1, 1, 0, 0, 0, 0, 0],  # a
            [1, 0, 0, 1, 1, 1, 0, 0],  # b
            [1, 0, 0, 0, 0, 0, 1, 1],  # c
            [0, 1, 0, 0, 0, 0, 0, 0],  # d
            [0, 1, 0, 0, 0, 0, 0, 0],  # e
            [0, 1, 0, 0, 0, 0, 0, 0],  # f
            [0, 0, 1, 0, 0, 0, 0, 0],  # g
            [0, 0, 1, 0, 0, 0, 0, 0]   # h
        ])
        self.assertTrue((adjacency == comparison).all())


    def test_valid_bfs_example_1(self):
        """
                       +---+
                       | a |
                       ++-++
                        | |
                +-------+ +------+
                |                |
              +-+-+            +-+-+
              | b |            | c |
              +++++            ++-++
               |||              | |
          +----+|+----+     +---+ +---+
          |     |     |     |         |
        +-+-+ +-+-+ +-+-+ +-+-+     +-+-+
        | d | | e | | f | | g |     | h |
        +---+ +---+ +---+ +---+     +---+
        """
        example_1 = t.tensor([
            #d  e  f  g  h
            [0, 1, 1, 2, 2], # d
            [1, 0, 1, 2, 2], # e
            [1, 1, 0, 2, 2], # f
            [2, 2, 2, 0, 1], # g
            [2, 2, 2, 1, 0]  # h
        ])
        adjacency = lca2adjacency(example_1, format='bfs')

        comparison = t.tensor([
            #a  b  c  d  e  f  g  h
            [0, 1, 1, 0, 0, 0, 0, 0],  # a
            [1, 0, 0, 1, 1, 1, 0, 0],  # b
            [1, 0, 0, 0, 0, 0, 1, 1],  # c
            [0, 1, 0, 0, 0, 0, 0, 0],  # d
            [0, 1, 0, 0, 0, 0, 0, 0],  # e
            [0, 1, 0, 0, 0, 0, 0, 0],  # f
            [0, 0, 1, 0, 0, 0, 0, 0],  # g
            [0, 0, 1, 0, 0, 0, 0, 0]   # h
        ])
        self.assertTrue((adjacency == comparison).all())

    def test_valid_bfs_example_2(self):
        """
                   +---+
                   | a |
                   ++-++
                    | |
             +------+ +------+
             |               |
             |             +-+-+
             |             | b |
             |             ++-++
             |              | |
             |           +--+ +---+
             |           |        |
           +-+-+       +-+-+      |
           | c |       | d |      |
           ++-++       ++-++      |
            | |         | |       |
          +-+ +-+     +-+ +-+     |
          |     |     |     |     |
        +-+-+ +-+-+ +-+-+ +-+-+ +-+-+
        | e | | f | | g | | h | | i |
        +---+ +---+ +---+ +---+ +---+
        """
        example_2 = t.tensor([
            #e  f  g  h  i
            [0, 1, 3, 3, 3],  # e
            [1, 0, 3, 3, 3],  # f
            [3, 3, 0, 1, 2],  # g
            [3, 3, 1, 0, 2],  # h
            [3, 3, 2, 2, 0]   # i
        ])
        adjacency = lca2adjacency(example_2)

        comparison = t.tensor([
            #a  b  c  d  e  f  g  h  i
            [0, 1, 1, 0, 0, 0, 0, 0, 0],  # a
            [1, 0, 0, 1, 0, 0, 0, 0, 1],  # b
            [1, 0, 0, 0, 1, 1, 0, 0, 0],  # c
            [0, 1, 0, 0, 0, 0, 1, 1, 0],  # d
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  # e
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  # f
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # g
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # h
            [0, 1, 0, 0, 0, 0, 0, 0, 0]   # i
        ])
        self.assertTrue((adjacency == comparison).all())

    def test_valid_bfs_example_3(self):
        """
              +---+
              | a |
              +++++
               |||
          +----+|+----+
          |     |     |
        +-+-+ +-+-+ +-+-+
        | b | | c | | d |
        +---+ +---+ +---+
        """
        example_3 = t.tensor([
            #b  c  d
            [0, 1, 1],  # b
            [1, 0, 1],  # c
            [1, 1, 0]   # d
        ])
        adjacency = lca2adjacency(example_3)

        comparison = t.tensor([
            #a  b  c  d
            [0, 1, 1, 1],  # a
            [1, 0, 0, 0],  # b
            [1, 0, 0, 0],  # c
            [1, 0, 0, 0]   # d
        ])
        self.assertTrue((adjacency == comparison).all())

    def test_valid_bfs_example_4(self):
        """
                         +---+
                         | a |
                         ++-++
                          | |
                     +----+ +----+
                     |           |
                   +-+-+         |
                   | b |         |
                   +++++         |
                    |||          |
            +-------+|+----+     |
            |        |     |     |
          +---+      |     |     |
          | c |      |     |     |
          ++-++      |     |     |
           | |       |     |     |
         +-+ +-+     |     |     |
         |     |     |     |     |
        +-+-+ +-+-+ +-+-+ +-+-+ +-+-+
        | d | | e | | f | | g | | h |
        +---+ +---+ +---+ +---+ +---+
        """
        example_4 = t.tensor([
            #g  h  d  f  c
            [0, 1, 2, 2, 3],  # g
            [1, 0, 2, 2, 3],  # h
            [2, 2, 0, 2, 3],  # d
            [2, 2, 2, 0, 3],  # f
            [3, 3, 3, 3, 0]   # c
        ])
        adjacency = lca2adjacency(example_4)

        comparison = t.tensor([
            #a  b  c  d  e  f  g  h
            [0, 1, 0, 0, 0, 0, 0, 1], #a
            [1, 0, 1, 0, 0, 1, 1, 0], #b
            [0, 1, 0, 1, 1, 0, 0, 0], #c
            [0, 0, 1, 0, 0, 0, 0, 0], #d
            [0, 0, 1, 0, 0, 0, 0, 0], #e
            [0, 1, 0, 0, 0, 0, 0, 0], #f
            [0, 1, 0, 0, 0, 0, 0, 0], #g
            [1, 0, 0, 0, 0, 0, 0, 0], #h
        ])
        self.assertTrue((adjacency == comparison).all())

    def test_valid_bfs_example_5(self):
        """
                       +---+
                       | a |
                       ++-++
                        | |
                 +------+ +------+
                 |               |
               +-+-+             |
               | b |             |
               ++-++             |
                | |              |
             +--+ +---+          |
             |        |          |
           +-+-+      |          |
           | d |      |          |
           ++-++      |          |
            | |       |         | |
          +-+ +-+     |       +-+ +-+
          |     |     |       |     |
        +-+-+ +-+-+ +-+-+   +-+-+ +-+-+
        | g | | h | | i |   | e | | f |
        +---+ +---+ +---+   +---+ +---+
        """
        example_5 = t.tensor([
            #e  f  g  h  i
            [0, 3, 3, 3, 3],  # e
            [3, 0, 3, 3, 3],  # f
            [3, 3, 0, 1, 2],  # g
            [3, 3, 1, 0, 2],  # h
            [3, 3, 2, 2, 0]   # i
        ])
        # Sort into bfs I think?
        ids = example_5.kthvalue(2, 0)[0].argsort()
        example_5 = example_5[ids][:, ids]
        adjacency = lca2adjacency(example_5)

        comparison = t.tensor([
            #a  b  d  g  h  i  e  f
            [0, 1, 0, 0, 0, 0, 1, 1],  # a
            [1, 0, 1, 0, 0, 1, 0, 0],  # b
            [0, 1, 0, 1, 1, 0, 0, 0],  # d
            [0, 0, 1, 0, 0, 0, 0, 0],  # g
            [0, 0, 1, 0, 0, 0, 0, 0],  # h
            [0, 1, 0, 0, 0, 0, 0, 0],  # i
            [1, 0, 0, 0, 0, 0, 0, 0],  # e
            [1, 0, 0, 0, 0, 0, 0, 0],  # f
        ])
        self.assertTrue((adjacency == comparison).all())


    # #########  DFS ###########
    def test_valid_dfs_example_1(self):
        """
                       +---+
                       | a |
                       ++-++
                        | |
                +-------+ +------+
                |                |
              +-+-+            +-+-+
              | b |            | f |
              +++++            ++-++
               |||              | |
          +----+|+----+     +---+ +---+
          |     |     |     |         |
        +-+-+ +-+-+ +-+-+ +-+-+     +-+-+
        | c | | d | | e | | g |     | h |
        +---+ +---+ +---+ +---+     +---+
        """
        example_1 = t.tensor([
            #c  d  e  g  h
            [0, 1, 1, 2, 2], # c
            [1, 0, 1, 2, 2], # d
            [1, 1, 0, 2, 2], # e
            [2, 2, 2, 0, 1], # g
            [2, 2, 2, 1, 0]  # h
        ])
        lca_matrix = lca2adjacency(example_1, format='dfs')

        comparison = t.tensor([
            #a  b  c  d  e  f  g  h
            [0, 1, 0, 0, 0, 1, 0, 0],  # a
            [1, 0, 1, 1, 1, 0, 0, 0],  # b
            [0, 1, 0, 0, 0, 0, 0, 0],  # c
            [0, 1, 0, 0, 0, 0, 0, 0],  # d
            [0, 1, 0, 0, 0, 0, 0, 0],  # e
            [1, 0, 0, 0, 0, 0, 1, 1],  # f
            [0, 0, 0, 0, 0, 1, 0, 0],  # g
            [0, 0, 0, 0, 0, 1, 0, 0]   # h
        ])
        self.assertTrue((lca_matrix == comparison).all())

    def test_valid_dfs_example_2(self):
        """
                   +---+
                   | a |
                   ++-++
                    | |
             +------+ +------+
             |               |
             |             +-+-+
             |             | e |
             |             ++-++
             |              | |
             |           +--+ +---+
             |           |        |
           +-+-+       +-+-+      |
           | b |       | f |      |
           ++-++       ++-++      |
            | |         | |       |
          +-+ +-+     +-+ +-+     |
          |     |     |     |     |
        +-+-+ +-+-+ +-+-+ +-+-+ +-+-+
        | c | | d | | g | | h | | i |
        +---+ +---+ +---+ +---+ +---+
        """
        example_2 = t.tensor([
            #c  d  g  h  i
            [0, 1, 3, 3, 3],  # c
            [1, 0, 3, 3, 3],  # d
            [3, 3, 0, 1, 2],  # g
            [3, 3, 1, 0, 2],  # h
            [3, 3, 2, 2, 0]   # i
        ])
        lca_matrix = lca2adjacency(example_2, format='dfs')

        comparison = t.tensor([
            #a  b  c  d  e  f  g  h  i
            [0, 1, 0, 0, 1, 0, 0, 0, 0],  # a
            [1, 0, 1, 1, 0, 0, 0, 0, 0],  # b
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # c
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # d
            [1, 0, 0, 0, 0, 1, 0, 0, 1],  # e
            [0, 0, 0, 0, 1, 0, 1, 1, 0],  # f
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # g
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # h
            [0, 0, 0, 0, 1, 0, 0, 0, 0]   # i
        ])
        self.assertTrue((lca_matrix == comparison).all())

    def test_valid_dfs_example_3(self):
        """
              +---+
              | a |
              +++++
               |||
          +----+|+----+
          |     |     |
        +-+-+ +-+-+ +-+-+
        | b | | c | | d |
        +---+ +---+ +---+
        """
        example_3 = t.tensor([
            #b  c  d
            [0, 1, 1], # b
            [1, 0, 1], # c
            [1, 1, 0]  # d
        ])
        lca_matrix = lca2adjacency(example_3, format='dfs')

        comparison = t.tensor([
            #a  b  c  d
            [0, 1, 1, 1], # a
            [1, 0, 0, 0], # b
            [1, 0, 0, 0], # c
            [1, 0, 0, 0]  # d
        ])
        self.assertTrue((lca_matrix == comparison).all())

    def test_valid_dfs_example_4(self):
        """
                         +---+
                         | a |
                         ++-++
                          | |
                     +----+ +----+
                     |           |
                   +-+-+         |
                   | b |         |
                   +++++         |
                    |||          |
            +-------+|+----+     |
            |        |     |     |
          +---+      |     |     |
          | c |      |     |     |
          ++-++      |     |     |
           | |       |     |     |
         +-+ +-+     |     |     |
         |     |     |     |     |
        +-+-+ +-+-+ +-+-+ +-+-+ +-+-+
        | d | | e | | f | | g | | h |
        +---+ +---+ +---+ +---+ +---+
        """
        example_4 = t.tensor([
            #d  e  f  g  h
            [0, 1, 2, 2, 3], # d
            [1, 0, 2, 2, 3], # e
            [2, 2, 0, 2, 3], # f
            [2, 2, 2, 0, 3], # g
            [3, 3, 3, 3, 0]  # h
        ])
        lca_matrix = lca2adjacency(example_4, format='dfs')

        comparison = t.tensor([
            #a  b  c  d  e  f  g  h
            [0, 1, 0, 0, 0, 0, 0, 1], # a
            [1, 0, 1, 0, 0, 1, 1, 0], # b
            [0, 1, 0, 1, 1, 0, 0, 0], # c
            [0, 0, 1, 0, 0, 0, 0, 0], # d
            [0, 0, 1, 0, 0, 0, 0, 0], # e
            [0, 1, 0, 0, 0, 0, 0, 0], # f
            [0, 1, 0, 0, 0, 0, 0, 0], # g
            [1, 0, 0, 0, 0, 0, 0, 0], # h
        ])
        self.assertTrue((lca_matrix == comparison).all())

    def test_valid_dfs_example_5(self):
        """
                    +---+
                    | a |
                    ++-++
                     | |
          +--------+-+ +-------+
          |        |           |
          |      +---+       +---+
          |      | b |       | c |
          |      ++-++       ++-++
          |       | |         | |
          |     +-+ +-+     +-+ +-+
          |     |     |     |     |
        +-+-+ +-+-+ +-+-+ +-+-+ +-+-+
        | d | | e | | f | | g | | h |
        +---+ +---+ +---+ +---+ +---+
        """
        example_5 = t.tensor([
            #d  e  f  g  h
            [0, 3, 3, 3, 3], # d
            [3, 0, 1, 3, 3], # e
            [3, 1, 0, 3, 3], # f
            [3, 3, 3, 0, 1], # g
            [3, 3, 3, 1, 0]  # h
        ])
        lca_matrix = lca2adjacency(example_5, format='dfs')

        comparison = t.tensor([
            #a  d  b  e  f  c  g  h
            [0, 1, 1, 0, 0, 1, 0, 0], # a
            [1, 0, 0, 0, 0, 0, 0, 0], # d
            [1, 0, 0, 1, 1, 0, 0, 0], # b
            [0, 0, 1, 0, 0, 0, 0, 0], # e
            [0, 0, 1, 0, 0, 0, 0, 0], # f
            [1, 0, 0, 0, 0, 0, 1, 1], # c
            [0, 0, 0, 0, 0, 1, 0, 0], # g
            [0, 0, 0, 0, 0, 1, 0, 0], # h
        ])
        self.assertTrue((lca_matrix == comparison).all())

    def test_illegal_empty_example(self):
        """
        +---+ +---+ +---+
        | a | | b | | c |
        +---+ +---+ +---+
        """
        zero_example = t.tensor([
            #a  b  c
            [0, 0, 0],  # a
            [0, 0, 0],  # b
            [0, 0, 0]   # c
        ])

        with self.assertRaises(InvalidLCAMatrix):
            lca2adjacency(zero_example, format='bfs')

    def test_illegal_floating_leaf(self):
        """
                 +---+
                 | b |
                 ++-++
                  | |
                +-+ +-+
                |     |
        +---+ +-+-+ +-+-+
        | a | | c | | d |
        +---+ +---+ +---+
        """
        floating_leaf_example = t.tensor([
            #a  b  c
            [0, 0, 0],  # a
            [0, 0, 1],  # b
            [0, 1, 0]   # c
        ])

        with self.assertRaises(InvalidLCAMatrix):
            lca2adjacency(floating_leaf_example)

    def test_illegal_asymmetric(self):
        """
        +---+ +---+ +---+
        | a | | b | | c |
        +---+ +---+ +---+
        """
        asymmetric_example = t.tensor([
            #a  b  c
            [0, 0, 0],  # a
            [1, 0, 0],  # b
            [0, 0, 0]   # c
        ])

        with self.assertRaises(InvalidLCAMatrix):
            lca2adjacency(asymmetric_example, format='bfs')

    def test_illegal_connections_1(self):
        illegal_connections_example = t.tensor([
            #a  b  c
            [0, 2, 1],  # a
            [2, 0, 1],  # b
            [1, 1, 0]   # c
        ])

        with self.assertRaises(InvalidLCAMatrix):
            lca2adjacency(illegal_connections_example, format='bfs')

    def test_illegal_connections_2(self):
        illegal_connections_example = t.tensor([
            #a  b  c
            [0, 1, 0],  # a
            [1, 0, 1],  # b
            [0, 1, 0]   # c
        ])

        with self.assertRaises(InvalidLCAMatrix):
            lca2adjacency(illegal_connections_example, format='bfs')

if __name__ == '__main__':
    unittest.main()

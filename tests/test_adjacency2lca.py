import torch as t
import unittest

from baumbauen.utils import adjacency2lca
from baumbauen.utils.adjacency2lca import InvalidAdjacencyMatrix


class AdjacencyReconstructionTest(unittest.TestCase):
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
        lca_matrix = adjacency2lca(example_1)

        comparison = t.tensor([
            #d  e  f  g  h
            [0, 1, 1, 2, 2], # d
            [1, 0, 1, 2, 2], # e
            [1, 1, 0, 2, 2], # f
            [2, 2, 2, 0, 1], # g
            [2, 2, 2, 1, 0]  # h
        ])
        self.assertTrue((lca_matrix == comparison).all())

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
        lca_matrix = adjacency2lca(example_2)

        comparison = t.tensor([
            #e  f  g  h  i
            [0, 1, 3, 3, 3],  # e
            [1, 0, 3, 3, 3],  # f
            [3, 3, 0, 1, 2],  # g
            [3, 3, 1, 0, 2],  # h
            [3, 3, 2, 2, 0]   # i
        ])
        self.assertTrue((lca_matrix == comparison).all())

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
            #a  b  c  d
            [0, 1, 1, 1],  # a
            [1, 0, 0, 0],  # b
            [1, 0, 0, 0],  # c
            [1, 0, 0, 0]   # d
        ])
        lca_matrix = adjacency2lca(example_3)

        comparison = t.tensor([
            #b  c  d
            [0, 1, 1],  # b
            [1, 0, 1],  # c
            [1, 1, 0]   # d
        ])
        self.assertTrue((lca_matrix == comparison).all())

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

        lca_matrix = adjacency2lca(example_4)

        comparison = t.tensor([
            #g  h  d  f  c
            [0, 1, 2, 2, 3],  # g
            [1, 0, 2, 2, 3],  # h
            [2, 2, 0, 2, 3],  # d
            [2, 2, 2, 0, 3],  # f
            [3, 3, 3, 3, 0]   # c
        ])
        self.assertTrue((lca_matrix == comparison).all())

    def test_valid_bfs_example_5(self):
        """
                 +---+
                 | a |
                 ++-++
                  | |
             +----+ +----+
             |           |
           +-+-+       +-+-+
           | b |       | c |
           +-+-+       ++-++
             |          | |
             |        +-+ +----+
             |        |        |
           +-+-+    +-+-+    +-+-+
           | d |    | e |    | f |
           ++-++    +-+-+    ++-++
            | |       |       | |
          +-+ +-+     |     +-+ +-+
          |     |     |     |     |
        +-+-+ +-+-+ +-+-+ +-+-+ +-+-+
        | g | | h | | i | | j | | k |
        +---+ +---+ +---+ +---+ +---+
        """
        example_5 = t.tensor([
            #a  b  c  d  e  f  g  h  i  j  k
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # a
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # b
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # c
            [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # d
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # e
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # f
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # g
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # h
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # i
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # j
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # k
        ])
        lca_matrix = adjacency2lca(example_5)

        comparison = t.tensor([
            #g  h  i  j  k
            [0, 1, 3, 3, 3],  # g
            [1, 0, 3, 3, 3],  # h
            [3, 3, 0, 2, 2],  # i
            [3, 3, 2, 0, 1],  # j
            [3, 3, 2, 1, 0]   # k
        ])
        self.assertTrue((lca_matrix == comparison).all())

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
        lca_matrix = adjacency2lca(example_1)

        comparison = t.tensor([
            #c  d  e  g  h
            [0, 1, 1, 2, 2], # c
            [1, 0, 1, 2, 2], # d
            [1, 1, 0, 2, 2], # e
            [2, 2, 2, 0, 1], # g
            [2, 2, 2, 1, 0]  # h
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
        lca_matrix = adjacency2lca(example_2)

        comparison = t.tensor([
            #c  d  g  h  i
            [0, 1, 3, 3, 3],  # c
            [1, 0, 3, 3, 3],  # d
            [3, 3, 0, 1, 2],  # g
            [3, 3, 1, 0, 2],  # h
            [3, 3, 2, 2, 0]   # i
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
            #a  b  c  d
            [0, 1, 1, 1], # a
            [1, 0, 0, 0], # b
            [1, 0, 0, 0], # c
            [1, 0, 0, 0]  # d
        ])
        lca_matrix = adjacency2lca(example_3)

        comparison = t.tensor([
            #b  c  d
            [0, 1, 1], # b
            [1, 0, 1], # c
            [1, 1, 0]  # d
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
        lca_matrix = adjacency2lca(example_4)

        comparison = t.tensor([
            #d  e  f  g  h
            [0, 1, 2, 2, 3], # d
            [1, 0, 2, 2, 3], # e
            [2, 2, 0, 2, 3], # f
            [2, 2, 2, 0, 3], # g
            [3, 3, 3, 3, 0]  # h
        ])
        self.assertTrue((lca_matrix == comparison).all())

    def test_valid_dfs_example_5(self):
        """
                 +---+
                 | a |
                 ++-++
                  | |
             +----+ +----+
             |           |
           +-+-+       +-+-+
           | b |       | f |
           +-+-+       ++-++
             |          | |
             |        +-+ +----+
             |        |        |
           +-+-+    +-+-+    +-+-+
           | c |    | g |    | i |
           ++-++    +-+-+    ++-++
            | |       |       | |
          +-+ +-+     |     +-+ +-+
          |     |     |     |     |
        +-+-+ +-+-+ +-+-+ +-+-+ +-+-+
        | d | | e | | h | | j | | k |
        +---+ +---+ +---+ +---+ +---+
        """
        example_5 = t.tensor([
            #a  b  c  d  e  f  g  h  i  j  k
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # a
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # b
            [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # c
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # d
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # e
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # f
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # g
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # h
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],  # i
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # j
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # k
        ])
        lca_matrix = adjacency2lca(example_5)

        comparison = t.tensor([
            #d  e  h  j  k
            [0, 1, 3, 3, 3], # d
            [1, 0, 3, 3, 3], # e
            [3, 3, 0, 2, 2], # h
            [3, 3, 2, 0, 1], # j
            [3, 3, 2, 1, 0]  # k
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

        with self.assertRaises(InvalidAdjacencyMatrix):
            adjacency2lca(zero_example)

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
            #a  b  c  d
            [0, 0, 0, 0],  # a
            [0, 0, 1, 1],  # b
            [0, 1, 0, 0],  # c
            [0, 1, 0, 0]   # d
        ])

        with self.assertRaises(InvalidAdjacencyMatrix):
            adjacency2lca(floating_leaf_example)

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

        with self.assertRaises(InvalidAdjacencyMatrix):
            adjacency2lca(asymmetric_example)

    def test_illegal_connections(self):
        illegal_connections_example = t.tensor([
            #a  b  c
            [0, 2, 1],  # a
            [2, 0, 1],  # b
            [1, 1, 0]   # c
        ])

        with self.assertRaises(InvalidAdjacencyMatrix):
            adjacency2lca(illegal_connections_example)


if __name__ == '__main__':
    unittest.main()

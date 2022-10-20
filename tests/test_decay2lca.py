import unittest
import torch as t
from phasespace import GenParticle

from baumbauen.utils import decay2lca


class LCAConstructionTest(unittest.TestCase):
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

        a = GenParticle('a', 15)
        b = GenParticle('b', 5)
        c = GenParticle('c', 5)
        d = GenParticle('d', 1)
        e = GenParticle('e', 1)
        f = GenParticle('f', 1)
        g = GenParticle('g', 1)
        h = GenParticle('h', 1)

        c.set_children(g, h)
        b.set_children(d, e, f)
        a.set_children(b, c)

        lca_matrix, names = decay2lca(a)
        comparison = t.tensor([
            #d  e  f  g  h
            [0, 1, 1, 2, 2], # d
            [1, 0, 1, 2, 2], # e
            [1, 1, 0, 2, 2], # f
            [2, 2, 2, 0, 1], # g
            [2, 2, 2, 1, 0]  # h
        ])
        self.assertTrue((t.Tensor(lca_matrix) == comparison).all())

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

        a = GenParticle('a', 20)
        b = GenParticle('b', 10)
        c = GenParticle('c', 5)
        d = GenParticle('d', 5)
        e = GenParticle('e', 1)
        f = GenParticle('f', 1)
        g = GenParticle('g', 1)
        h = GenParticle('h', 1)
        i = GenParticle('i', 1)

        c.set_children(e, f)
        b.set_children(d, i)
        d.set_children(g, h)
        a.set_children(c, b)

        lca_matrix, names = decay2lca(a)

        comparison = t.tensor([
            #e  f  g  h  i
            [0, 1, 3, 3, 3],  # e
            [1, 0, 3, 3, 3],  # f
            [3, 3, 0, 1, 2],  # g
            [3, 3, 1, 0, 2],  # h
            [3, 3, 2, 2, 0]   # i
        ])
        self.assertTrue((t.Tensor(lca_matrix) == comparison).all())

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
        a = GenParticle('a', 20)
        b = GenParticle('b', 5)
        c = GenParticle('c', 5)
        d = GenParticle('d', 5)

        a.set_children(b, c, d)
        lca_matrix, names = decay2lca(a)

        comparison = t.tensor([
            #b  c  d
            [0, 1, 1],  # b
            [1, 0, 1],  # c
            [1, 1, 0]   # d
        ])
        self.assertTrue((t.Tensor(lca_matrix) == comparison).all())

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

        a = GenParticle('a', 20)
        b = GenParticle('b', 10)
        c = GenParticle('c', 5)
        d = GenParticle('d', 1)
        e = GenParticle('e', 1)
        f = GenParticle('f', 1)
        g = GenParticle('g', 1)
        h = GenParticle('h', 1)

        c.set_children(d, e)
        b.set_children(c, f, g)
        a.set_children(b, h)

        lca_matrix, names = decay2lca(a)

        comparison = t.tensor([
            #g  h  d  f  c
            [0, 1, 2, 2, 3],  # g
            [1, 0, 2, 2, 3],  # h
            [2, 2, 0, 2, 3],  # d
            [2, 2, 2, 0, 3],  # f
            [3, 3, 3, 3, 0]   # c
        ])
        self.assertTrue((t.Tensor(lca_matrix) == comparison).all())

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
        a = GenParticle('a', 20)
        b = GenParticle('b', 5)
        c = GenParticle('c', 1)
        d = GenParticle('d', 1)
        e = GenParticle('e', 1)
        f = GenParticle('f', 5)
        g = GenParticle('g', 1)
        h = GenParticle('h', 1)

        b.set_children(c, d, e)
        f.set_children(g, h)
        a.set_children(b, f)

        lca_matrix, names = decay2lca(a)
        comparison = t.tensor([
            #c  d  e  g  h
            [0, 1, 1, 2, 2], # c
            [1, 0, 1, 2, 2], # d
            [1, 1, 0, 2, 2], # e
            [2, 2, 2, 0, 1], # g
            [2, 2, 2, 1, 0]  # h
        ])
        self.assertTrue((t.Tensor(lca_matrix) == comparison).all())

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
        a = GenParticle('a', 20)
        b = GenParticle('b', 5)
        c = GenParticle('c', 1)
        d = GenParticle('d', 1)
        e = GenParticle('e', 10)
        f = GenParticle('f', 5)
        g = GenParticle('g', 1)
        h = GenParticle('h', 1)
        i = GenParticle('i', 1)

        b.set_children(c, d)
        f.set_children(g, h)
        e.set_children(f, i)
        a.set_children(b, e)

        lca_matrix, names = decay2lca(a)
        comparison = t.tensor([
            #c  d  g  h  i
            [0, 1, 3, 3, 3],  # c
            [1, 0, 3, 3, 3],  # d
            [3, 3, 0, 1, 2],  # g
            [3, 3, 1, 0, 2],  # h
            [3, 3, 2, 2, 0]   # i
        ])

        self.assertTrue((t.Tensor(lca_matrix) == comparison).all())

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
        a = GenParticle('a', 10)
        b = GenParticle('b', 1)
        c = GenParticle('c', 1)
        d = GenParticle('d', 1)

        a.set_children(b, c, d)

        lca_matrix, names = decay2lca(a)
        comparison = t.tensor([
            #b  c  d
            [0, 1, 1], # b
            [1, 0, 1], # c
            [1, 1, 0]  # d
        ])
        self.assertTrue((t.Tensor(lca_matrix) == comparison).all())

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

        a = GenParticle('a', 20)
        b = GenParticle('b', 10)
        c = GenParticle('c', 5)
        d = GenParticle('d', 1)
        e = GenParticle('e', 1)
        f = GenParticle('f', 1)
        g = GenParticle('g', 1)
        h = GenParticle('h', 1)

        c.set_children(d, e)
        b.set_children(c, f, g)
        a.set_children(b, h)

        lca_matrix, names = decay2lca(a)

        comparison = t.tensor([
            #d  e  f  g  h
            [0, 1, 2, 2, 3], # d
            [1, 0, 2, 2, 3], # e
            [2, 2, 0, 2, 3], # f
            [2, 2, 2, 0, 3], # g
            [3, 3, 3, 3, 0]  # h
        ])
        self.assertTrue((t.Tensor(lca_matrix) == comparison).all())


    def test_valid_example_5(self):
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
           +-+-+       +-+-+    +-+-+
           | c |       | d |    | e |
           ++-++       ++-++    ++-++
        """

        a = GenParticle('a', 10)
        b = GenParticle('b', 5)
        c = GenParticle('c', 1)
        d = GenParticle('d', 1)
        e = GenParticle('e', 1)

        b.set_children(d, e)
        a.set_children(c, b)

        lca_matrix, names = decay2lca(a)

        comparison = t.tensor([
            #c  d  e
            [0, 2, 2], # c
            [2, 0, 1], # d
            [2, 1, 0]  # e
        ])
        self.assertTrue((t.Tensor(lca_matrix) == comparison).all())

    def test_valid_example_6(self):
        # TODO draw properly
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
           +-+-+       +-+-+    +-+-+
           | c |       | d |    | e |
           ++-++       ++-++    ++-++
        """

        a = GenParticle('a', 30)
        b = GenParticle('b', 15)
        c = GenParticle('c', 10)
        d = GenParticle('d', 10)

        e = GenParticle('e', 5)
        f = GenParticle('f', 5)

        g = GenParticle('g', 1)
        h = GenParticle('h', 1)
        i = GenParticle('i', 1)
        j = GenParticle('j', 1)
        k = GenParticle('k', 1)
        l = GenParticle('l', 1)
        m = GenParticle('m', 1)

        e.set_children(h, i)
        f.set_children(k, l)
        c.set_children(g, e)
        d.set_children(j, f)
        b.set_children(d, m)
        a.set_children(c, b)

        lca_matrix, names = decay2lca(a)

        comparison = t.tensor([
            #g, h, i, j, k, l, m,
            [0, 2, 2, 4, 4, 4, 4], # g
            [2, 0, 1, 4, 4, 4, 4], # h
            [2, 1, 0, 4, 4, 4, 4], # i
            [4, 4, 4, 0, 2, 2, 3], # j
            [4, 4, 4, 2, 0, 1, 3], # k
            [4, 4, 4, 2, 1, 0, 3], # l
            [4, 4, 4, 3, 3, 3, 0], # m
        ])
        self.assertTrue((t.Tensor(lca_matrix) == comparison).all())

if __name__ == '__main__':
    unittest.main()

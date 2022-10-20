import unittest
from phasespace import GenParticle

from baumbauen.utils import is_isomorphic_decay, assign_parenthetical_weight_tuples

class DecayIsomorphismTest(unittest.TestCase):
    def test_example_1(self):
        a = GenParticle('a', 10)
        b = GenParticle('b', 1)
        c = GenParticle('c', 5)
        d = GenParticle('d', 1)
        e = GenParticle('e', 1)

        c.set_children(d, e)
        a.set_children(b, c)

        A = GenParticle('a', 10)
        B = GenParticle('b', 1)
        C = GenParticle('c', 5)
        D = GenParticle('d', 1)
        E = GenParticle('e', 1)

        C.set_children(E, D)
        A.set_children(C, B)

        self.assertTrue(is_isomorphic_decay(a, A))

    def test_example_2(self):
        a = GenParticle('a', 10)
        b = GenParticle('b', 1)
        c = GenParticle('c', 5)
        d = GenParticle('d', 1)
        e = GenParticle('e', 1)

        c.set_children(d, e)
        a.set_children(b, c)

        A = GenParticle('a', 10)
        B = GenParticle('b', 2)
        C = GenParticle('c', 5)
        D = GenParticle('d', 1)
        E = GenParticle('e', 1)

        C.set_children(E, D)
        A.set_children(C, B)

        self.assertFalse(is_isomorphic_decay(a, A))

    def test_example_3(self):
        a = GenParticle('a', 10)
        b = GenParticle('b', 1)
        c = GenParticle('c', 5)

        a.set_children(b, c)

        A = GenParticle('a', 10)
        B = GenParticle('b', 1)
        C = GenParticle('c', 5)
        D = GenParticle('d', 1)
        E = GenParticle('e', 1)

        C.set_children(E, D)
        A.set_children(C, B)

        self.assertFalse(is_isomorphic_decay(a, A))


if __name__ == '__main__':
    unittest.main()

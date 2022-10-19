
def is_isomorphic_decay(root1, root2):
    t1 = assign_parenthetical_weight_tuples(root1)
    t2 = assign_parenthetical_weight_tuples(root2)

    if t1 == t2:
        return True
    return False


def assign_parenthetical_weight_tuples(node):
    """
    Args:
        node(phasespace.GenParticle

    After A. Aho, J. Hopcroft, and J. Ullman The Design and Analysis of Computer Algorithms. Addison-Wesley Publishing Co., Reading, MA, 1974, pp. 84-85.
    Credits to Alexander Smal

    """
    if not node.has_children:
        return f'({node.get_mass()})'

    child_tuples = [assign_parenthetical_weight_tuples(c) for c in node.children]
    child_tuples.sort()
    child_tuples = ''.join(child_tuples)

    return f'({node.get_mass()}{child_tuples})'

from phasespace import GenParticle


def generate_decay_tree(num_left_children, num_right_children, root_mass, left_mass, right_mass, leaf_mass):

    total_nparticles = num_left_children + num_right_children
    leaf_list = []
    for i in range(total_nparticles):
        leaf_list.append(GenParticle('leaf{}'.format(i), leaf_mass))
    left = GenParticle('left', left_mass).set_children(*leaf_list[:num_left_children])
    right = GenParticle('right', right_mass).set_children(*leaf_list[num_left_children:])
    root = GenParticle('root', root_mass).set_children(left, right)
    return left, right, root

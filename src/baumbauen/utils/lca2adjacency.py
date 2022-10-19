import torch as t

from .tree_utils import is_valid_tree


class InvalidLCAMatrix(Exception):
    """
    Specialized Exception sub-class raised for malformed LCA-gram-matrices or LCA-gram-matrices that do not encode trees.
    """
    pass


class Node:
    def __init__(self, level, children):
        self.level = level
        self.children = children

        self.parent = None
        self.bfs_index = -1
        self.dfs_index = -1


def _get_ancestor(node):
    """
    Trail search for the highest ancestor of a node.

    Args:
        node (Node): A node instance for which to determine the ancestor.

    Returns:
        Node: the node's ancestor, returns self if a disconnected leaf node.
    """
    ancestor = node

    while ancestor.parent is not None:
        ancestor = ancestor.parent

    return ancestor


def _pull_down(node):
    """
    Work up the node's history, pulling down a level any nodes
    whose children are all more than one level below.

    Performs the operation in place

    Args:
        node (Node): A node instance for which to determine the ancestor.

    Returns:
        Node: the node's ancestor, returns self if a disconnected leaf node.
    """
    # First check the children
    if len(node.children) > 0:
        highest_child = max([c.level for c in node.children])
        node.level = highest_child + 1

    # Then move on to the parent
    if node.parent is not None:
        _pull_down(node.parent)

    return


def _breadth_first_enumeration(root, queue, adjacency_matrix):
    """
    Enumerates the tree breadth-first into a queue

    Args:
        root (Node): the root of the tree to get the enumeration of.
        queue (dict): level-indexed dictionary
        adjacency_matrix (iterable): 2-dimensional LCA-gram-matrix (M, M).
    """
    # insert current root node into the queue
    level = root.level
    queue.setdefault(level, []).append(root)

    # enumerate the children
    for child in root.children:
        _breadth_first_enumeration(child, queue, adjacency_matrix)

    return queue


def _breadth_first_adjacency(root, adjacency_matrix):
    """
    Enumerates the tree breadth-first into a queue

    Args:
        root (Node): the root of the tree to get the enumeration of.
        queue (dict): level-indexed dictionary
        adjacency_matrix (iterable): 2-dimensional LCA-gram-matrix (M, M).
    """
    queue = _breadth_first_enumeration(root, {}, adjacency_matrix)

    # on recursion end in the root node, traverse the tree once to assign bfs ids to each node
    index = 0
    for i in range(root.level, 0, -1):
        for node in queue[i]:
            node.bfs_index = index
            index += 1

    # then traverse the tree again to fill in the adjacencies
    for i in range(root.level, 0, -1):
        for node in queue[i]:
            for child in node.children:
                adjacency_matrix[node.bfs_index, child.bfs_index] = 1
                adjacency_matrix[child.bfs_index, node.bfs_index] = 1


def _depth_first_enumeration(root, index):
    """
    Fills the passed adjacency matrix based on the tree spanned by `root`.

    Args:
        root (Node): the root of the tree to enumerate the indices of.
        index (int): 2-dimensional adjacency matrix to be filled (N, N).
    """
    root.dfs_index = index
    branch_index = index

    for child in root.children:
        branch_index = _depth_first_enumeration(child, branch_index + 1)

    return branch_index


def _depth_first_labeling(root, adjacency_matrix):
    """
    Sets the adjacencies for the a node and its children depth-first.

    Args:
        root (Node): the root of the tree to get the adjacency matrix of.
        lca_matrix (iterable): 2-dimensional adjacency matrix to be filled (N, N).
    """
    for child in root.children:
        adjacency_matrix[child.dfs_index, root.dfs_index] = 1
        adjacency_matrix[root.dfs_index, child.dfs_index] = 1

        _depth_first_labeling(child, adjacency_matrix)


def _depth_first_adjacency(root, adjacency_matrix):
    """
    Fills the passed adjacency matrix based on the tree spanned by `root`.

    Args:
        root (Node): the root of the tree to get the adjacency matrix of.
        lca_matrix (iterable): 2-dimensional adjacency matrix to be filled (N, N).
    """
    _depth_first_enumeration(root, index=0)
    _depth_first_labeling(root, adjacency_matrix)


def _reconstruct(lca_matrix):
    """
    Does the actual heavy lifting of the adjacency matrix reconstruction. Traverses the LCA matrix level-by-level,
    starting at one. For each level new nodes have to be inserted into the adjacency matrix, if a LCA matrix with this
    level number exists. The newly created node(s) will then be connected to the lower leaves, respectively,
    sub-graphs. This function may produce reconstructions that are valid graphs, but not trees.

    Args:
        lca_matrix (iterable): 2-dimensional LCA-gram-matrix (M, M).

    Returns:
        Node: the root node of the reconstructed tree.
        int: the total number of nodes in the tree

    Raises:
        InvalidLCAMatrix: If passed LCA-gram-matrix is not a tree.
    """
    n = lca_matrix.shape[0]
    total_nodes = n
    # depths = int(lca_matrix.max())
    levels = sorted(lca_matrix.unique().tolist())
    # Want to skip over leaves
    levels.remove(0)

    # create nodes for all leaves
    leaves = [Node(1, []) for _ in range(n)]

    # iterate level-by-level through the matrix, starting from immediate connections
    # we can correct missing intermediate levels here too
    # Just use current_level to check the actual LCA entry, once we know which level it is
    # (ignoring missed levels) then use the index (corrected level)
    # for current_level in range(1, depths + 1):
    for idx, current_level in enumerate(levels, 1):
        # iterate through each leaf in the LCA matrix
        for column in range(n):
            # iterate through all corresponding nodes
            # the LCA matrix is symmetric, hence, check only the from the diagonal down
            for row in range(column + 1, n):
                # skip over entries not in current level
                if lca_matrix[row, column] <= 0:
                    raise InvalidLCAMatrix
                elif lca_matrix[row, column] != current_level:
                    continue

                # get the nodes
                a_node = leaves[column]
                another_node = leaves[row]

                # determine the ancestors of both nodes
                an_ancestor = _get_ancestor(a_node)
                a_level = an_ancestor.level

                another_ancestor = _get_ancestor(another_node)
                another_level = another_ancestor.level

                # The nodes both already have an ancestor at that level, confirm it's the same one
                if a_level == another_level == (idx + 1):
                    if an_ancestor is not another_ancestor:
                        raise InvalidLCAMatrix
                # Should also check neither have an ancestor above the current level
                # If so then something went really wrong
                elif a_level > idx + 1 or another_level > idx + 1:
                    raise InvalidLCAMatrix

                # The nodes don't have an ancestor at the level we're inspecting.
                # We need to make one and connect them to it
                elif a_level < idx + 1 and another_level < idx + 1:
                    # parent = Node(max(a_level, another_level) + 1, [an_ancestor, another_ancestor])
                    parent = Node(idx + 1, [an_ancestor, another_ancestor])
                    an_ancestor.parent = parent
                    another_ancestor.parent = parent
                    total_nodes += 1

                # the left node already has a higher order parent, lets attach to it
                # I think should confirm that a_level == idx + 1 too
                elif another_level < idx + 1 and a_level == idx + 1:
                    # This should be the another_ancestor.parent getting assigned
                    # another_node.parent = an_ancestor
                    # an_ancestor.children.append(another_node)
                    another_ancestor.parent = an_ancestor
                    an_ancestor.children.append(another_ancestor)

                # same for right
                elif a_level < idx + 1 and another_level == idx + 1:
                    an_ancestor.parent = another_ancestor
                    another_ancestor.children.append(an_ancestor)

                # If all this fails I think that's also bad
                else:
                    raise InvalidLCAMatrix

    # The LCAs aren't guaranteed to actually be "lowest" ancestors, we need to make sure
    # by pulling down any nodes that can be (i.e. have all children more than one level down)
    for leaf in leaves:
        _pull_down(leaf)

    # we have created the tree structure, let's initialize the adjacency matrix and find the root to traverse from
    root = _get_ancestor(leaves[0])

    return root, total_nodes


def lca2adjacency(lca_matrix, format='bfs'):
    """
    Converts a tree's LCA-gram matrix representation, i.e. a square matrix (M, M), where each row/column corresponds to
    a leaf of the tree and each matrix entry is the level of the lowest-common-ancestor (LCA) of the two leaves, into
    the corresponding two-dimension adjacency matrix (N,N), with M < N. The levels are enumerated top-down from the
    root.

    Args:
        lca_matrix (iterable): 2-dimensional LCA-gram-matrix (M, M).
        format (string): output format of the generated adjacency matrix. Can be either on of the two 'bfs' or 'dfs' for
            bread-first or depth-first.

    Returns:
        iterable: 2-dimensional matrix (N, N) encoding the graph's node adjacencies. Linked nodes have values unequal to
            zero.

    Raises:
        InvalidLCAMatrix: If passed LCA-gram-matrix is malformed (e.g. not 2d or not square) or does not encode a tree.
    """
    # sanitize the output format
    if format not in {'bfs', 'dfs'}:
        raise ValueError(f'format must be one of bfs|dfs, but was {format}')

    # ensure input is torch tensor or can be converted to it
    if not isinstance(lca_matrix, t.Tensor):
        try:
            lca_matrix = t.Tensor(lca_matrix)
        except TypeError as err:
            print(f'Input type must be compatible with torch Tensor: {err}')
            raise

    # ensure two dimensions
    if len(lca_matrix.shape) != 2:
        raise InvalidLCAMatrix

    # ensure that it is square
    n, m = lca_matrix.shape
    if n != m:
        raise InvalidLCAMatrix

    # check symmetry
    if not (lca_matrix == lca_matrix.T).all():
        raise InvalidLCAMatrix

    try:
        root, total_nodes = _reconstruct(lca_matrix)
    except IndexError:
        raise InvalidLCAMatrix

    # allocate the adjacency matrix
    adjacency_matrix = t.zeros((total_nodes, total_nodes), dtype=t.int64)
    try:
        if format == 'bfs':
            _breadth_first_adjacency(root, adjacency_matrix)
        else:
            _depth_first_adjacency(root, adjacency_matrix)
    except IndexError:
        raise InvalidLCAMatrix

    # check whether what we reconstructed is actually a tree - might be a regular graph for example
    if not is_valid_tree(adjacency_matrix):
        raise InvalidLCAMatrix

    return adjacency_matrix

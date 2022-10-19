import numpy as np


def shuffle_together(arr, mat):

    """ Shuffle the array along the second axis and the matrix and vec correspondingly
    Args:
        arr (numpy.ndarray): N dimensional array, e.g. (B, M, F)
        mat (numpy.ndarray): An MxM matrix, where M is the same size as the axis being shuffled in arr
        vec (numpy.ndarray): An one-dimensional array of length M (deprecated)

    Returns:
        tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray): Returns the triple of arr shuffled
        along the second axis (B, M, F),
        axis of mat shuffled similarly (B, M, M) and vec shuffeled (B, M)
    """
    """ Ensure the axis sizes match """
    assert arr.shape[1] == mat.shape[0]
    assert arr.shape[1] == mat.shape[1]

    """ Create the array of output shuffled arrays """
    shuff_arr = np.zeros(arr.shape)
    shuff_mat = np.zeros((arr.shape[0], *mat.shape))

    dim_size = arr.shape[1]
    """ Row by row shuffle the input arr """
    for idx in np.arange(arr.shape[0]):
        perms = np.random.permutation(dim_size)
        shuff_arr[idx] = arr[idx, perms]
        shuff_mat[idx] = mat[perms][:, perms]

    return shuff_arr, shuff_mat

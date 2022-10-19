import torch as t


def ordinalise_labels(labels, n_classes):
    ''' Encode integer labels as softmax of ordinal euclidean distance from correspoinding one-hot position.

    Values along the first dimension of the output are the softmaxed euclidean distance to the
    arg position indicated by the corresponding integer in the labels input.
    E.g. [[0, 1, 2]], with n_classes=3, returns (columns sum to one)
    [[[0.0900, 0.4223, 0.6652],
    [0.2447, 0.1554, 0.2447],
    [0.6652, 0.4223, 0.0900]]]

    Inputs:
        labels (torch.Tensor): Shape (N, d_0, d_1, ..., d_K)
        n_classes (int): Number of classes
    Output:
        (torch.Tensor): Shape (N, n_classes, d_0, d_1, ..., d_K)
    '''

    # Need to know the dims of the input labels create our ordinal matrix
    labels_dims = labels.ndim

    # Create the distances to calculate from
    # This produces repeated aranges for every batch entry and each of the dims of the labels
    dist_mat = t.arange(n_classes)
    dist_mat = dist_mat.reshape(1, n_classes, *[1] * (labels_dims - 1))  # (1, n_classes, 1, ..., 1)
    dist_mat = dist_mat.expand(labels.size()[0], n_classes, *list(labels.size())[1:])  # (N, n_classes, d_0, ..., d_K)

    # And subtract the labels from the aranges along the classes dimension to get the distances
    # The labels[:, None] tells pytorch to subtract along the class dimension
    dist_mat = t.abs(dist_mat.sub(labels[:, None]))  # (N, n_classes, d_0, ..., d_K)

    # The final piece is to softmax the inverse, this makes the argmax at the position of the true label in the class dimension
    return t.softmax(-dist_mat, 1)

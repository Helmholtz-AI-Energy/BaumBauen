import torch
import numpy as np


def default_collate_fn(batch):

    input, target = list(zip(*batch))
    input = torch.stack(input)
    target = torch.stack(target)
    input = input.transpose(0, 1)
    return input, target


def pad_collate_fn(batch):
    ''' Collate function for batches with varying sized inputs

    This pads the batch with zeros to the size of the large sample in the batch

    Args:
        batch(tuple):  batch contains a list of tuples of structure (sequence, target)
    Return:
        (tuple): Input, labels, mask, all padded
    '''
    # First pad the input data
    data = [item[0] for item in batch]
    # Here we pad with 0 as it's the input, so need to indicate that the network ignores it
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)  # (N, L, F)
    data = data.transpose(0, 1)  # (L, N, F)
    # Then the labels
    labels = [item[1] for item in batch]

    # Note the -1 padding, this is where we tell the loss to ignore the outputs in those cells
    target = torch.zeros(data.shape[1], data.shape[0], data.shape[0], dtype=torch.long) - 1  # (N, L, L)
    # mask = torch.zeros(data.shape[0], data.shape[1], data.shape[1])  # (N, L, L)

    # I don't know a cleaner way to do this, just copying data into the fixed-sized tensor
    for i, tensor in enumerate(labels):
        length = tensor.size(0)
        target[i, :length, :length] = tensor
        # mask[i, :length, :length] = 1

    return data, target  # mask


def rel_pad_collate_fn(batch, self_interaction=False):
    ''' Collate function for batches with varying sized inputs

    This pads the batch with zeros to the size of the large sample in the batch

    Args:
        batch(tuple):  batch contains a list of tuples of structure (sequence, target)
    Return:
        (tuple): Input, labels, rel_rec, rel_send, all padded
    '''
    lens = [sample[0].size(0) for sample in batch]

    data, target = pad_collate_fn(batch)

    rel_recvs = construct_rel_recvs(lens, self_interaction=self_interaction)
    rel_sends = construct_rel_sends(lens, self_interaction=self_interaction)

    return (data, rel_recvs, rel_sends), target


def construct_rel_recvs(ln_leaves, self_interaction=False, device=None):
    """
    ln_leaves: list of ints, number of leaves for each sample in the batch
    """
    pad_len = max(ln_leaves)
    rel_recvs = []
    for l in ln_leaves:
        rel_recv = torch.eye(pad_len, device=device)  # (l, l)
        rel_recv[:, l:] = 0
        rel_recv = rel_recv.repeat_interleave(pad_len, dim=1).T  # (l*l, l)
        for j in range(l, pad_len):  # remove padding vertex edges TODO optimize
            rel_recv[j::pad_len] = 0

        if self_interaction is False:
            rel_recv[0::pad_len + 1] = 0

        rel_recvs.append(rel_recv)

    return torch.stack(rel_recvs)


def construct_rel_sends(ln_leaves, self_interaction=False, device=None):
    """
    ln_leaves: list of ints, number of leaves for each sample in the batch
    """
    pad_len = max(ln_leaves)
    rel_sends = []
    for l in ln_leaves:
        rel_send = torch.eye(pad_len, device=device).repeat(pad_len, 1)
        if self_interaction is False:
            rel_send[torch.arange(0, pad_len * pad_len, pad_len + 1)] = 0
            # rel_send = rel_send[rel_send.sum(dim=1) > 0]  # (l*l, l)

        # padding
        rel_send[:, l:] = 0
        rel_send[l * (pad_len):] = 0
        rel_sends.append(rel_send)
    return torch.stack(rel_sends)


def calculate_class_weights(dataloader, num_classes, num_batches=100, amp_enabled=False):
    """ Calculates class weights based on num_batches of the dataloader

    This assumes there exists a -1 padding value that is not part of the class weights.
    Any classes not found will have a weight of one set

    Args:
        dataloader(torch.Dataloader): Dataloader to iterate through when collecting batches
        num_classes(int): Number of classes
        num_batches(int, optional): Number of batches from dataloader to use to approximate class weights
        amp_enabled(bool, optional): Enabled mixed precision. Creates weights tensor as half precision
    Return:
        (torch.tensor): Tensor of class weights, normalised to 1
    """
    weights = torch.zeros((num_classes,))
    for i, batch in zip(range(num_batches), dataloader):
        index, count = torch.unique(batch[1], sorted=True, return_counts=True)
        # TODO: add padding value as input to specifically ignore
        if -1 in index:
            # This line here assumes that the lowest class found is -1 (padding) which should be ignored
            weights[index[1:]] += count[1:]
        else:
            weights[index] += count

    # The weights need to be the invers, since we scale down the most common classes
    weights = 1 / weights
    # Set inf to 1
    weights = torch.nan_to_num(weights, posinf=float('nan'))
    # And normalise to sum to 1
    weights = weights / weights.nansum()
    # Finally, assign default value to any that were missing during calculation time
    weights = torch.nan_to_num(weights, nan=1)

    return weights


def pull_down_LCA(lca):
    ''' Generic method to pull down a single LCA

    This assumes that all LCA values are positive, if it contains e.g. padding those will be also pulled down.
    '''

    # Only support a single LCA at a time
    assert lca.ndim == 2, 'LCA to pull down must be a single LCA with 2 dimensions'

    # Do for torch tensors
    if isinstance(lca, torch.Tensor):
        unique_els = torch.unique(lca, sorted=True)
        for idx, val in enumerate(unique_els):
            lca[lca == val] = idx

    # And for numpy arrays
    elif isinstance(lca, np.ndarray):
        unique_els = np.unique(lca)
        for idx, val in np.ndenumerate(unique_els):
            lca[lca == val] = idx

    # Raise error if anything else
    else:
        raise TypeError('Only torch.tensor or numpy.ndarray supported for LCA pull-down')

    return lca

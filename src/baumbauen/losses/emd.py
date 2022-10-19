#!/usr/bin/env python3

import torch
from torch import nn


class EMDLoss(nn.Module):
    r''' Earth Mover's distance loss.

    Takes the l-norm distance and calculates Earth Mover's Distance in one dimension.
    Inputs are same format as to PyTorch's CrossEntropyLoss.

    From the Mallow's distance equivalence this is defined as
    .. math::
        \text{EMD}(\hat{y}, y) = \left(\frac{1}{C}\right)^{\frac{1}{p}}
        \left\lVert \text{CDF}(\hat{y}) - \text{CDF}(y) \right\rVert_l

    Default behaviour is to take the L2 norm squared.

    Args:
        p (int, optional): The p-norm distance used to calculate EMD.
            Default: 2
        take_root (bool, optional): Whether to take the p-th root of the norm.
            This is generally not done to speed up convergence.
            Default: False
        normalise (bool, optional): Whether to normalise the p-norm by the
            :math:`\left(\frac{1}{C}\right)` factor or ignore it. This is
            recommended when dealing with variable sized inputs as larger inputs
            will generate larger norms. Default: True

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss. This should be of type LongTensor.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.
    '''
    def __init__(self, l=2, take_root=False, normalise=True):
        ''' Maybe add reduce input here? '''
        super(EMDLoss, self).__init__()
        self.l = l
        self.take_root = take_root
        self.normalise = normalise

    def forward(self, input, target):
        # First need to one-hot encode the targets

        one_hot_target = torch.zeros(input.size(), dtype=torch.long)  # (N, C, d1, ..., dK)
        # Channels are first dimension (first arg), populating the index 'target' with ones (last arg)
        one_hot_target = one_hot_target.scatter(1, target.unsqueeze(1), 1)  # (N, C, d1, ..., dK)

        # From definition of EMD, need to take the diference in CDFs of target and input
        cdf_target = torch.cumsum(one_hot_target, 1)  # (N, C, d1, ..., dK)
        cdf_input = torch.cumsum(input, 1)  # (N, C, d1, ..., dK)

        # Take the norm
        norm = torch.sum(torch.pow(torch.abs(cdf_input - cdf_target), self.l), 1)  # (N, d1, ..., dK)

        if self.normalise:
            norm /= input.size()[2]  # (N, d1, ..., dK)
        if self.take_root:
            norm = torch.pow(norm, 1. / self.l)

        # Should add reduce as a parameter, setting to mean for now
        return torch.mean(norm)

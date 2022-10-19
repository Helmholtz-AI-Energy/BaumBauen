#!/usr/bin/env python3

import torch
from torch import nn


class OuterSumLayer(nn.Module):
    """ Outer summation

    Expands an S=(l, b, c) input to matrix L=(l, l, b, c) by setting elements L_{i,j}
    in L to the sum of S_i and S_j.
    """

    def __init__(self):
        super(OuterSumLayer, self).__init__()

    def forward(self, x):
        """
        Input of shape (l, b, c)
        Output of shape (l, l, b, c)
        """
        l, b, c = x.size()

        x = x.unsqueeze(1)  # l, 1, b, c
        x = x.expand(l, l, b, c)
        x = (x + torch.transpose(x, 0, 1)).contiguous()  # l, l, b, c

        return x


class OuterProductLayer(nn.Module):
    ''' Outer product

    Expands S=(l, b, c) input to output matrix L=(l, l, b, c) by setting elements L_{i,j}
    in L to the product of S_i and S_j.
    Essentially just the attention map in a self attention layer.
    '''

    def __init__(self):
        super(OuterProductLayer, self).__init__()

    def forward(self, x):
        """
        Inputs of shape (l, b, c)
        Output of shape (l, l, b, c)
        """
        return torch.einsum('lbc,mbc->lmbc', x, x)  # l, l, b, c


class CatLayer(nn.Module):
    """ Outer concatenation with weights """

    def __init__(self, d):
        super(CatLayer, self).__init__()
        self.f = nn.LeakyReLU(0.1)
        self.l = nn.Linear(2 * d, d)

        return

    def forward(self, x):
        """
        input of shape (l, b, c)
        output of shape (l*l, b, c)
        """
        x = self._outer_concat(x)
        x = self.f(x)
        x = self.l(x)  # l^2, b, c
        return x

    def _outer_concat(self, x):
        """
        input of shape (l, b, c)
        output of shape (l*l, b, 2*c)

        """
        l, b, c = x.size()

        x = x.unsqueeze(1)  # l, 1, b, c
        x = x.expand(l, l, b, c)
        x = torch.cat((x, torch.transpose(x, 0, 1)), -1).contiguous()  # l, l, b, 2*c
        return x.reshape(-1, b, 2 * c)


# Could combine this with the above, but the output shape would change.
class OuterConcatLayer(nn.Module):
    """ Outer concatenation without weights

    Stacks input pairs in new dimension.
    """

    def __init__(self):
        super(OuterConcatLayer, self).__init__()
        return

    def forward(self, x, y=None):
        """
        Shape:
            - Input: (l, N, C)
            - Output: (l, l, N, 2C)
        """
        x = self._outer_concat(x, y)  # l, l, N, 2C
        return x

    def _outer_concat(self, x, y=None):
        """
        Shape:
            - Input: (l, N, C)
            - Output: (l, l, N, 2C)
        """
        l, N, C = x.size()

        if y is None:
            x = x.unsqueeze(1)  # l, 1, N, C
            x = x.expand(l, l, N, C)  # l, 1, N, C
            x = torch.cat(
                (x, torch.transpose(x, 0, 1)), -1
            ).contiguous()  # l, l, N, 2*C
        else:
            l_y = y.size(0)

            x = x.unsqueeze(1)
            x = x.expand(l, l_y, N, C)
            y = y.unsqueeze(0)
            y = y.expand(l, l_y, N, C)

            x = torch.cat((x, y), -1).contiguous()
        return x


class SymmetrizeLayer(nn.Module):
    """ Symmetrize an input matrix.

    This does not preserve any softmax on the input matrix.

    Inputs:
        - mode (str): "mean" or "max". Specifies symmetrization method performed.
    """

    def __init__(self, mode="mean"):
        super(SymmetrizeLayer, self).__init__()

        assert mode in ["mean", "max"], ValueError("mode must be one of mean or max")
        self.mode = mode

        return

    def forward(self, x):
        """
        Shape:
            - Input: (l, l, N, C)
            - Output: (l, l, N, C)
        """
        if self.mode == "mean":
            return torch.div(torch.add(x, torch.transpose(x, 0, 1)), 2)
        elif self.mode == "max":
            return torch.max(x, torch.transpose(x, 0, 1))


class DiagonalizeLayer(nn.Module):
    """ Diagonalize an one-hot encoded input matrix :

    Sets a large positive value for the zeroth class of all the diagonal terms.

    Sets the additive inverse of this value to all the other classes
    """

    def __init__(self, nodes, nclass):
        super(DiagonalizeLayer, self).__init__()
        self.nodes = nodes
        self.nclass = nclass
        return

    def forward(self, x):
        """ Creates a tensor with 20 for the zeroth class of diagonal elements

        and -20 for the other classes.

        This tensor is added to the input matrix.

        Shape:
            - Input: (n, N, C)
            - Output: (n, N, C)

        """

        device = x.device
        zdiag = torch.zeros([self.nodes * self.nodes, self.nclass], device=device)
        i, j = 1, 0

        while i <= self.nodes * self.nodes:
            zdiag[i, 0] = 20
            zdiag[i, 1:self.nclass] = -20
            i = j
            i = i * (self.nodes + 1)
            j += 1

        x = x + zdiag
        return x

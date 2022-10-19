import torch
from torch import nn


class FocalLoss(nn.Module):
    ''' Non weighted version of Focal Loss

    This is a blend of:
        https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
        https://amaarora.github.io/2020/06/29/FocalLoss.html
    '''
    def __init__(self, gamma=2, ignore_index=0, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):

        CE_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            weight=self.weight,
        )
        targets = targets.type(torch.long)

        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt)**self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

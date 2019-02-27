import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gama=2., size_average=True, weight=None):
        super(FocalLoss, self).__init__()
        '''
        weight: size(C)
        '''
        self.gama = gama
        self.size_average = size_average
        self.weight = weight
    def forward(self, inputs, targets):
        '''
        inputs: size(N,C)
        targets: size(N)
        '''
        log_P = -F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        P = torch.exp(log_P)
        batch_loss = -torch.pow(1-P, self.gama).mul(log_P)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, inputs, targets):
        mul = torch.mul(inputs, targets)
        add = torch.add(inputs, 1, targets)
        dice = 2 * torch.div(mul.sum(), add.sum())
        return 1 - dice

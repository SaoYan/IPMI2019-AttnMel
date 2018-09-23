import torch
import torch.nn as nn
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     def __init__(self, gama=2., size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gama = gama
#         self.size_average = size_average
#     def forward(self, inputs, targets):
#         '''
#         inputs: size(N,C), float tensor
#         targets: size(N,C), one-hot, float tensor
#         '''
#         N, C = inputs.size()
#         P = F.softmax(inputs, dim=1).mul(targets).sum(dim=1) # (N)
#         log_P = F.log_softmax(inputs, dim=1).mul(targets).sum(dim=1) # (N)
#         batch_loss = -torch.pow(1-P, self.gama).mul(log_P)
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss

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

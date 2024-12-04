# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import torch
from torch.nn import functional as F

"""refer to https://github.com/yatengLG/Focal-Loss-Pytorch/blob/master/Focal_Loss.py"""

class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 3, reduction='mean'):
        super(focal_loss,self).__init__()
        self.reduction = reduction
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
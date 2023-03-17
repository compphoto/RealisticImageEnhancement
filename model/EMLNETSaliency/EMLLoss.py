import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.eps = 1e-6

    def KL_loss(self, input, target):
        input = input / input.sum() 
        target = target / target.sum()
        loss = (target * torch.log(target/(input+self.eps) + self.eps)).sum()
        return loss 

    def CC_loss(self, input, target):
        input = (input - input.mean()) / input.std()  
        target = (target - target.mean()) / target.std()
        loss = (input * target).sum() / (torch.sqrt((input*input).sum() * (target * target).sum()))
        loss = 1 - loss
        return loss

    def NSS_loss(self, input, target):
        ref = (target - target.mean()) / target.std()
        input = (input - input.mean()) / input.std()
        loss = (ref*target - input*target).sum() / target.sum()
        return loss 

    def forward(self, input, fix, smap):
        kl = 0
        cc = 0
        nss = 0
        for p, f, s in zip(input, fix, smap):
            kl += self.KL_loss(p, s)
            cc += self.CC_loss(p, s)
            nss += self.NSS_loss(p, f)
        return (kl + cc + nss) / input.size(0)

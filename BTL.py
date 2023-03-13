import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BTL(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BTL, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel)

    def forward(self, feature):
        feature = self.fc(feature)
        feature = F.normalize(feature, dim=1)
        feature = SimSignGrad.apply(feature)
        return feature
    
class SimSignGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_x):
        return grad_x
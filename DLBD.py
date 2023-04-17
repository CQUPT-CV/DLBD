import torch.nn as nn
import torchvision.models as models

from BTL import BTL

class DLBD(nn.Module):
    def __init__(self, bit, pretrained=True):
        super(DLBD, self).__init__()
        model_conv = models.resnet18(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(model_conv.children())[:-1])
        self.BTL = BTL(512, bit)

    def forward(self, img_pair):
        feature = self.cnn(img_pair)
        feature = feature.squeeze()
        bin = self.BTL(feature)
        return bin
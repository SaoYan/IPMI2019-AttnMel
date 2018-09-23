import torch
import torch.nn as nn
import torch.nn.functional as F
from initialize import *
import torchvision.models as models

class AttnResNet(nn.Module):
    def __init__(self, num_classes, attention=True, normalize_attn=True, init='kaimingNormal'):
        super(AttnResNet, self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.net.fc = nn.Linear(2048, num_classes)
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self.net.fc)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self.net.fc)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self.net.fc)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self.net.fc)
        else:
            raise NotImplementedError("Invalid type of initialization!")
    def forward(self, x):
        c1, c2, c3 = None, None, None
        out = self.net(x)
        return [out, c1, c2, c3]

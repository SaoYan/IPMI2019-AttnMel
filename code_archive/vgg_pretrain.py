import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, SelfAttentionBlock
from initialize import *
import torchvision.models as models

'''
non-local self-attention
'''

class AttnVGG(nn.Module):
    def __init__(self, num_classes, attention=True, normalize_attn=True, init='kaimingNormal'):
        super(AttnVGG, self).__init__()
        self.attention = attention
        self.net = models.vgg16_bn(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self.net.classifier)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self.net.classifier)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self.net.classifier)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self.net.classifier)
        else:
            raise NotImplementedError("Invalid type of initialization!")
    def forward(self, x):
        out = self.net(x)
        c1, c2, c3 = None, None, None
        return [out, c1, c2, c3]

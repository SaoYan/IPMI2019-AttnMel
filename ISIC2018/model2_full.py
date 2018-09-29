import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, LinearAttentionBlock, ProjectorBlock
from initialize import *

'''
complete VGG16
'''

class AttnVGG(nn.Module):
    def __init__(self, num_classes, attention=True, normalize_attn=True, init='kaimingNormal'):
        super(AttnVGG, self).__init__()
        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.classify = nn.Sequential(
            nn.Linear(in_features=512*8*8, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")
    def forward(self, x):
        # feed forward
        block1 = F.max_pool2d(self.conv_block1(x), kernel_size=2, stride=2) # /2
        block2 = F.max_pool2d(self.conv_block2(block1), kernel_size=2, stride=2) # /4
        l1 = F.max_pool2d(self.conv_block3(block2), kernel_size=2, stride=2) # /8
        l2 = F.max_pool2d(self.conv_block4(l1), kernel_size=2, stride=2) # /16
        l3 = F.max_pool2d(self.conv_block5(l2), kernel_size=2, stride=2) # /32, batch_sizex512x8x8
        N, __, __, __ = l3.size()
        l3 = l3.view(N, -1)
        c1, c2, c3 = None, None, None
        out = self.classify(l3)
        return [out, c1, c2, c3]

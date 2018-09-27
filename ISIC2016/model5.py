import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, SelfAttentionBlock
from initialize import *

'''
non-local self-attention
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
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True),
            nn.AdaptiveAvgPool2d(output_size=(1,1)) # gloal average pooling
        )
        # Projectors & Compatibility functions
        if self.attention:
            #self.attn1 = SelfAttentionBlock(in_features=64, attn_features=32, subsample=True, mode='gaussian')
            self.attn2 = SelfAttentionBlock(in_features=128, attn_features=64, subsample=True, mode='gaussian')
            self.attn3 = SelfAttentionBlock(in_features=256, attn_features=128, subsample=True, mode='gaussian')
            self.attn4 = SelfAttentionBlock(in_features=512, attn_features=256, subsample=True, mode='gaussian')
        # final classification layer
        self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
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
        if self.attention:
            block1 = F.max_pool2d(self.conv_block1(x), kernel_size=2, stride=2) # /2
            block2 = self.attn2(F.max_pool2d(self.conv_block2(block1), kernel_size=2, stride=2)) # /4 
            block3 = self.attn3(F.max_pool2d(self.conv_block3(block2), kernel_size=2, stride=2)) # /8
            block4 = self.attn4(F.max_pool2d(self.conv_block4(block3), kernel_size=2, stride=2)) # /16
            block5 = F.max_pool2d(self.conv_block5(block4), kernel_size=2, stride=2) # /32
        else:
            block1 = F.max_pool2d(self.conv_block1(x), kernel_size=2, stride=2) # /2
            block2 = F.max_pool2d(self.conv_block2(block1), kernel_size=2, stride=2) # /4
            block3 = F.max_pool2d(self.conv_block3(block2), kernel_size=2, stride=2) # /8
            block4 = F.max_pool2d(self.conv_block4(block3), kernel_size=2, stride=2) # /16
            block5 = F.max_pool2d(self.conv_block5(block4), kernel_size=2, stride=2) # /32
        g = self.feature(block5) # /32 --> batch_sizex512x1x1
        out = self.classify(torch.squeeze(g))
        c1, c2, c3 = None, None, None
        return [out, c1, c2, c3]

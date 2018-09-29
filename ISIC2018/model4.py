import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, GridAttentionBlock
from initialize import *

'''
Grid attention
Pay attention after max-pooling
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
            self.attn1 = GridAttentionBlock(in_features_l=256, in_features_g=512, attn_features=512, up_factor=4, normalize_attn=normalize_attn, output_transform=False)
            self.attn2 = GridAttentionBlock(in_features_l=512, in_features_g=512, attn_features=512, up_factor=2, normalize_attn=normalize_attn, output_transform=False)
        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=256+512*2, out_features=num_classes, bias=True)
        else:
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
        # modification
        if self.attention:
            if normalize_attn:
                nn.init.constant_(self.attn1.phi.bias, 10.0)
            else:
                nn.init.constant_(self.attn1.phi.bias, 3.0)
    def forward(self, x):
        # feed forward
        block1 = F.max_pool2d(self.conv_block1(x), kernel_size=2, stride=2) # /2
        block2 = F.max_pool2d(self.conv_block2(block1), kernel_size=2, stride=2) # /4
        l1 = F.max_pool2d(self.conv_block3(block2), kernel_size=2, stride=2) # /8
        l2 = F.max_pool2d(self.conv_block4(l1), kernel_size=2, stride=2) # /16
        g = F.max_pool2d(self.conv_block5(l2), kernel_size=2, stride=2) # /32
        pooled = self.feature(g).squeeze()
        # pay attention
        if self.attention:
            c1, f1 = self.attn1(l1, g)
            c2, f2 = self.attn2(l2, g)
            c3 = None
            f = torch.cat((pooled,f1,f2), dim=1) # batch_sizexC
            # classification layer
            out = self.classify(f) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            out = self.classify(pooled)
        return [out, c1, c2, c3]

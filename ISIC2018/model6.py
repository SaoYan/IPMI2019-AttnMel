import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ResNetBottleneckBlock, LinearAttentionBlock, ProjectorBlock
from initialize import *

'''
Linear attention
'''

class AttnRes164(nn.Module):
    def __init__(self, num_classes, attention=True, normalize_attn=True, init='kaimingNormal'):
        super(AttnRes164, self).__init__()
        self.attention = attention
        features = [16, 64, 128, 256]
        n_blocks = [18, 18, 18]
        # conv block
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=features[0], kernel_size=7, stride=2, padding=3, bias=False), # /2
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # /4
        )
        # res block1
        self.layer1 = self._make_layer(ResNetBottleneckBlock, in_features=features[0], out_features=features[1], num_blocks=n_blocks[0], stride=2) # 256, /8
        self.layer2 = self._make_layer(ResNetBottleneckBlock, in_features=features[1], out_features=features[2], num_blocks=n_blocks[1], stride=2) # 512, /16
        self.layer3 = self._make_layer(ResNetBottleneckBlock, in_features=features[2], out_features=features[3], num_blocks=n_blocks[2], stride=2) # 1024, /32
        self.feature = nn.Sequential(
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1,1)) # global average pooling
        )
        # Projectors & Compatibility functions
        if self.attention:
            self.projector1 = ProjectorBlock(features[1], 256)
            self.projector2 = ProjectorBlock(features[2], 256)
            self.attn1 = LinearAttentionBlock(256, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(256, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(256, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=3*256, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Sequential(
                nn.Linear(in_features=features[3], out_features=512, bias=True),
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
    def _make_layer(self, block, in_features, out_features, num_blocks, stride=1):
        layers = []
        layers.append(block(in_features=in_features, out_features=out_features, stride=stride)) # /2
        for i in range(1, num_blocks):
            layers.append(block(in_features=out_features, out_features=out_features, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        # feed forward
        pre = self.pre_conv(x) # /4
        l1 = self.layer1(pre) # /8
        l2 = self.layer2(l1) # /16
        l3 = self.layer3(l2) # /32
        g = self.feature(l3) # /32 --> batch_sizex256x1x1
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector1(l1), g)
            c2, g2 = self.attn2(self.projector2(l2), g)
            c3, g3 = self.attn3(l3, g)
            g_hat = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            # classification layer
            out = self.classify(g_hat) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            out = self.classify(torch.squeeze(g))
        return [out, c1, c2, c3]

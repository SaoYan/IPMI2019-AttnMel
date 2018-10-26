import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import SelfAttentionBlock
import torchvision.models as models

'''
vgg
pretrain
non-local self-attention
'''

class AttnVGG(nn.Module):
    def __init__(self, num_classes, attention=True, normalize_attn=True):
        super(AttnVGG, self).__init__()
        self.attention = attention
        net = models.vgg16_bn(pretrained=True)
        self.conv_block1 = nn.Sequential(*list(net.features.children())[0:7])
        self.conv_block2 = nn.Sequential(*list(net.features.children())[7:14])
        self.conv_block3 = nn.Sequential(*list(net.features.children())[14:24])
        self.conv_block4 = nn.Sequential(*list(net.features.children())[24:34])
        self.conv_block5 = nn.Sequential(*list(net.features.children())[34:44])
        self.dense = nn.Sequential(*list(net.classifier.children())[:-1])
        self.classify = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        if self.attention:
            self.attn2 = SelfAttentionBlock(in_features=128, attn_features=64, subsample=True)
            self.attn3 = SelfAttentionBlock(in_features=256, attn_features=128, subsample=True)
            self.attn4 = SelfAttentionBlock(in_features=512, attn_features=256, subsample=True)
            self.attn5 = SelfAttentionBlock(in_features=512, attn_features=256, subsample=True)
        # initialize
        self.reset_parameters(self.classify)
        if self.attention:
            self.reset_parameters(self.attn2)
            self.reset_parameters(self.attn3)
            self.reset_parameters(self.attn4)
            self.reset_parameters(self.attn5)
    def reset_parameters(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        if self.attention:
            block1 = self.conv_block1(x)# /2
            block2 = self.attn2(self.conv_block2(block1)) # /4
            block3 = self.attn3(self.conv_block3(block2)) # /8
            block4 = self.attn4(self.conv_block4(block3)) # /16
            block5 = self.attn5(self.conv_block5(block4)) # /32
        else:
            block1 = self.conv_block1(x) # /2
            block2 = self.conv_block2(block1) # /4
            block3 = self.conv_block3(block2) # /8
            block4 = self.conv_block4(block3) # /16
            block5 = self.conv_block5(block4) # /32
        N, __, __, __ = block5.size()
        fc = self.dense(block5.view(N,-1))
        out = self.classify(fc)
        c1, c2, c3 = None, None, None
        return [out, c1, c2, c3]

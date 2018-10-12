import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import SelfAttentionBlock
import torchvision.models as models

'''
resnet v1
pretrain
non-local self-attention
'''

class AttnResNet(nn.Module):
    def __init__(self, num_classes, attention=True, normalize_attn=True, init='kaimingNormal'):
        super(AttnResNet, self).__init__()
        self.attention = attention
        net = models.resnet50(pretrained=True)
        self.pre_conv = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool) # /4
        self.layer1 = net.layer1 # /4
        self.layer2 = net.layer2 # /8
        self.layer3 = net.layer3 # /16
        self.layer4 = net.layer4 # /32
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classify = nn.Linear(2048, num_classes)
        if self.attention:
            self.attn1 = SelfAttentionBlock(in_features=256, attn_features=128, subsample=True)
            self.attn2 = SelfAttentionBlock(in_features=512, attn_features=256, subsample=True)
            self.attn3 = SelfAttentionBlock(in_features=1024, attn_features=512, subsample=True)
            self.attn4 = SelfAttentionBlock(in_features=2048, attn_features=1024, subsample=True)
        # initialize
        self.reset_parameters(self.classify)
        if self.attention:
            self.reset_parameters(self.attn1)
            self.reset_parameters(self.attn2)
            self.reset_parameters(self.attn3)
            self.reset_parameters(self.attn4)
    def reset_parameters(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        pre = self.pre_conv(x)       # /4
        if self.attention:
            layer1 = self.attn1(self.layer1(pre))    # /4
            layer2 = self.attn2(self.layer2(layer1)) # /8
            layer3 = self.attn3(self.layer3(layer2)) # /16
            layer4 = self.attn4(self.layer4(layer3)) # /32
        else:
            layer1 = self.layer1(pre)    # /4
            layer2 = self.layer2(layer1) # /8
            layer3 = self.layer3(layer2) # /16
            layer4 = self.layer4(layer3) # /32
        pool = self.avgpool(layer4)
        N, __, __, __ = pool.size()
        out = self.classify(pool.view(N,-1))
        c1, c2, c3 = None, None, None
        return [out, c1, c2, c3]

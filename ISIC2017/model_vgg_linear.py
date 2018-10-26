import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import LinearAttentionBlock
import torchvision.models as models

'''
vgg
pretrain
linear attention
'''

class AttnVGG(nn.Module):
    def __init__(self, num_classes, attention=True, normalize_attn=False):
        super(AttnVGG, self).__init__()
        self.attention = attention
        net = models.vgg16_bn(pretrained=True)
        self.conv_block1 = nn.Sequential(*list(net.features.children())[0:7])
        self.conv_block2 = nn.Sequential(*list(net.features.children())[7:14])
        self.conv_block3 = nn.Sequential(*list(net.features.children())[14:24])
        self.conv_block4 = nn.Sequential(*list(net.features.children())[24:34])
        self.conv_block5 = nn.Sequential(*list(net.features.children())[34:44])
        if self.attention:
            self.dense = nn.Linear(in_features=512*7*7, out_features=512, bias=True)
            self.attn1 = LinearAttentionBlock(512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(512, normalize_attn=normalize_attn)
            self.classify = nn.Linear(in_features=512*3, out_features=num_classes, bias=True)
        else:
            self.dense = nn.Sequential(*list(net.classifier.children())[:-1])
            self.classify = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        # initialize
        self.reset_parameters(self.classify)
        if self.attention:
            self.reset_parameters(self.dense)
            self.reset_parameters(self.attn1)
            self.reset_parameters(self.attn2)
    def reset_parameters(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        block1 = self.conv_block1(x) # /2
        block2 = self.conv_block2(block1) # /4
        block3 = self.conv_block3(block2) # /8
        block4 = self.conv_block4(block3) # /16
        block5 = self.conv_block5(block4) # /32
        N, __, __, __ = block5.size()
        g = self.dense(block5.view(N,-1))
        if self.attention:
            c1, g1 = self.attn1(block4, g.view(N,512,1,1))
            c2, g2 = self.attn2(block5, g.view(N,512,1,1))
            g_hat = torch.cat((g,g1,g2), dim=1) # batch_size x C
            out = self.classify(g_hat)
        else:
            out = self.classify(g)
            c1, c2 = None, None
        return [out, None, c1, c2]

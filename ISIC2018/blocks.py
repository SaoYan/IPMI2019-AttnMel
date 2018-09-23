import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

class ResNetBottleneckBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super(ResNetBottleneckBlock, self).__init__()
        self.downsample = None
        res_features = int(out_features / 4)
        self.res = nn.Sequential(
            nn.BatchNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, res_features, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(res_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_features, res_features, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(res_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_features, out_features, kernel_size=1, padding=0, bias=False),
        )
        #  projection shortcut
        if (stride != 1) or (in_features != out_features):
            self.downsample = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=stride, bias=False)
    def forward(self, x):
        residual = self.res(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = F.sigmoid(c)
        g = torch.mul(a.expand_as(l), l).view(N,C,-1).sum(dim=2) # batch_sizexC
        return c.view(N,1,W,H), g

class GridAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True, output_transform=False):
        super(GridAttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.output_transform = output_transform
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
        if self.output_transform:
            self.trans = nn.Sequential(
                nn.Conv2d(in_channels=in_features_l, out_channels=in_features_l, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(in_features_l)
            )
    def forward(self, l, g):
        N, C, W, H = l.size()
        g = self.W_g(g)
        up_g = F.interpolate(g, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(self.W_l(l) + up_g)) # batch_sizex1xWxH
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = F.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.output_transform:
            f = self.trans(f)
        return c.view(N,1,W,H), f.view(N,C,-1).sum(dim=2)

class SelfAttentionBlock(nn.Module):
    def __init__(self, transform=False, normalize_attn=True, in_features=None, attn_features=None):
        super(SelfAttentionBlock, self).__init__()
        self.transform = transform
        self.normalize_attn = normalize_attn
        if self.transform:
            self.theta   = nn.Conv2d(in_channels=in_features, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
            self.phi     = nn.Conv2d(in_channels=in_features, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
            self.g       = nn.Conv2d(in_channels=in_features, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
            self.restore = nn.Conv2d(in_channels=attn_features, out_channels=in_features, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        N, C, W, H = x.size()
        # transform
        if self.transform:
            x1 = self.theta(x).view(N,C,-1).permute((0,2,1)) # N x WH x C
            x2 = self.phi(x).view(N,C,-1)                    # N x C X WH
            x3 = self.g(x).view(N,C,-1)                      # N x C X WH
        else:
            x1 = x.view(N,C,-1).permute((0,2,1)) # N x WH x C
            x2 = x.view(N,C,-1)                  # N x C X WH
            x3 = x.view(N,C,-1)                  # N x C X WH
        # attention
        c = torch.bmm(x1, x2)
        if self.normalize_attn:
            attn = x3.bmm(F.softmax(c.view(N,-1),dim=1).view(N,W*H,W*H)).view(N,C,W,H)
        else:
            attn = x3.bmm(F.sigmoid(c)).view(N,C,W,H)
        # restore feature
        if self.transform:
            attn = self.restore(attn)
        return c, x + attn

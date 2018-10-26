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
            layers.append(nn.ReLU(inplace=True))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

'''
ResNet v1 & v2 bottleneck block

Reference Papers:
Deep Residual Learning for Image Recognition https://arxiv.org/pdf/1512.03385.pdf
Identity Mappings in Deep Residual Networks https://arxiv.org/pdf/1603.05027.pdf

Reference code:
ResNet v1 & v2 https://github.com/facebook/fb.resnet.torch
ResNet v1 https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
class ResNetBottleneckBlock(nn.Module):
    def __init__(self, in_features, out_features, pre_act=False, first_stage=False, stride=1, factor=4, normalize_attn=None):
        '''
        normalize_attn is a redundant parameter!!
        when pre_act & first_stage are both True: no preact for the first conv (see paper for details)
        when pre_act is False, first_stage is ignored
        '''
        super(ResNetBottleneckBlock, self).__init__()
        self.downsample = None
        self.pre_act = pre_act
        self.first_stage = first_stage
        res_features = out_features // factor
        # ops
        if self.pre_act:
            if not self.first_stage:
                self.bn1   = nn.BatchNorm2d(in_features, affine=True, track_running_stats=True)
            self.conv1 = nn.Conv2d(in_features, res_features, kernel_size=1, padding=0, bias=False)
            self.bn2   = nn.BatchNorm2d(res_features, affine=True, track_running_stats=True)
            self.conv2 = nn.Conv2d(res_features, res_features, kernel_size=3, padding=1, stride=stride, bias=False)
            self.bn3   = nn.BatchNorm2d(res_features, affine=True, track_running_stats=True)
            self.conv3 = nn.Conv2d(res_features, out_features, kernel_size=1, padding=0, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_features, res_features, kernel_size=1, padding=0, bias=False)
            self.bn1   = nn.BatchNorm2d(res_features, affine=True, track_running_stats=True)
            self.conv2 = nn.Conv2d(res_features, res_features, kernel_size=3, padding=1, stride=stride, bias=False)
            self.bn2   = nn.BatchNorm2d(res_features, affine=True, track_running_stats=True)
            self.conv3 = nn.Conv2d(res_features, out_features, kernel_size=1, padding=0, bias=False)
            self.bn3   = nn.BatchNorm2d(out_features, affine=True, track_running_stats=True)
        #  projection shortcut
        if (stride != 1) or (in_features != out_features):
            self.downsample = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=stride, bias=False)
            if not self.pre_act:
                self.downsample = nn.Sequential(
                    self.downsample,
                    nn.BatchNorm2d(out_features, affine=True, track_running_stats=True)
                )
    def forward(self, x):
        if self.pre_act:
            if self.first_stage:
                pre = x
            else:
                pre = F.relu(self.bn1(x), inplace=True)
            if self.downsample is not None:
                shortcut = self.downsample(pre) # the shortcut also act in a pre-activation way!!!
            else:
                shortcut = x
            conv1 = self.conv1(pre)
            conv2 = self.conv2(F.relu(self.bn2(conv1), inplace=True))
            conv3 = self.conv3(F.relu(self.bn3(conv2), inplace=True))
            out = conv3 + shortcut
        else:
            conv1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
            conv2 = F.relu(self.bn2(self.conv2(conv1)), inplace=True)
            conv3 = self.bn3(self.conv3(conv2))
            if self.downsample is not None:
                shortcut = self.downsample(x)
            else:
                shortcut = x
            out = F.relu(conv3+shortcut, inplace=True)
        return out

'''
ResNet v1 & v2 bottleneck block + CBAM attention

Reference Papers:
Deep Residual Learning for Image Recognition https://arxiv.org/pdf/1512.03385.pdf
Identity Mappings in Deep Residual Networks https://arxiv.org/pdf/1603.05027.pdf
Convolutional Block Attention Module https://arxiv.org/pdf/1807.06521.pdf

Reference code:
ResNet v1 & v2 https://github.com/facebook/fb.resnet.torch
ResNet v1 https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
CBAM: https://github.com/kobiso/CBAM-keras
'''
class ResNetBottleneckBlock_CBAM(nn.Module):
    def __init__(self, in_features, out_features, pre_act=False, first_stage=False, stride=1, factor=4, normalize_attn=False):
        '''
        when pre_act & first_stage are both True: no preact for the first conv (see paper for details)
        when pre_act is False, first_stage is ignored
        '''
        super(ResNetBottleneckBlock_CBAM, self).__init__()
        self.downsample = None
        self.pre_act = pre_act
        self.first_stage = first_stage
        res_features = out_features // factor
        # ops
        if self.pre_act:
            if not self.first_stage:
                self.bn1   = nn.BatchNorm2d(in_features, affine=True, track_running_stats=True)
            self.conv1 = nn.Conv2d(in_features, res_features, kernel_size=1, padding=0, bias=False)
            self.bn2   = nn.BatchNorm2d(res_features, affine=True, track_running_stats=True)
            self.conv2 = nn.Conv2d(res_features, res_features, kernel_size=3, padding=1, stride=stride, bias=False)
            self.bn3   = nn.BatchNorm2d(res_features, affine=True, track_running_stats=True)
            self.conv3 = nn.Conv2d(res_features, out_features, kernel_size=1, padding=0, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_features, res_features, kernel_size=1, padding=0, bias=False)
            self.bn1   = nn.BatchNorm2d(res_features, affine=True, track_running_stats=True)
            self.conv2 = nn.Conv2d(res_features, res_features, kernel_size=3, padding=1, stride=stride, bias=False)
            self.bn2   = nn.BatchNorm2d(res_features, affine=True, track_running_stats=True)
            self.conv3 = nn.Conv2d(res_features, out_features, kernel_size=1, padding=0, bias=False)
            self.bn3   = nn.BatchNorm2d(out_features, affine=True, track_running_stats=True)
        # attention block
        self.cbam = CBAMAttentionBlock(out_features, reduction=16, normalize_attn=normalize_attn)
        #  projection shortcut
        if (stride != 1) or (in_features != out_features):
            self.downsample = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=stride, bias=False)
            if not self.pre_act:
                self.downsample = nn.Sequential(
                    self.downsample,
                    nn.BatchNorm2d(out_features, affine=True, track_running_stats=True)
                )
    def forward(self, x):
        if self.pre_act:
            if self.first_stage:
                pre = x
            else:
                pre = F.relu(self.bn1(x), inplace=True)
            if self.downsample is not None:
                shortcut = self.downsample(pre) # the shortcut also act in a pre-activation way!!!
            else:
                shortcut = x
            conv1 = self.conv1(pre)
            conv2 = self.conv2(F.relu(self.bn2(conv1), inplace=True))
            conv3 = self.conv3(F.relu(self.bn3(conv2), inplace=True))
            __, cbam = self.cbam(conv3)
            out = cbam + shortcut
        else:
            conv1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
            conv2 = F.relu(self.bn2(self.conv2(conv1)), inplace=True)
            conv3 = self.bn3(self.conv3(conv2))
            if self.downsample is not None:
                shortcut = self.downsample(x)
            else:
                shortcut = x
            cbam = self.cbam(conv3)
            out = F.relu(cbam+shortcut, inplace=True)
        return out

'''
Reference paper
Learn To Pay Attention https://arxiv.org/abs/1804.02391

Reference code
https://github.com/DadianisBidza/LearnToPayAttention-Keras
'''
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
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l).view(N,C,-1).sum(dim=2) # batch_sizexC
        return c.view(N,1,W,H), g

class NonLinearAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, normalize_attn=True):
        super(NonLinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=True)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.phi( F.relu(self.W_l(l) + self.W_g(g)))
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g

'''
Grid attention block

Reference papers
Attention-Gated Networks https://arxiv.org/abs/1804.05338 & https://arxiv.org/abs/1808.08114

Reference code
https://github.com/DadianisBidza/LearnToPayAttention-Keras
'''
class GridAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True):
        super(GridAttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C)
        return c.view(N,1,W,H), output

'''
Self attention

Reference paper
Non-local Neural Networks https://arxiv.org/abs/1711.07971

Reference code
caffe2  https://github.com/facebookresearch/video-nonlocal-net
pytorch https://github.com/AlexHex7/Non-local_pytorch
keras   https://github.com/titu1994/keras-non-local-nets
'''
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_features, attn_features, subsample=True):
        super(SelfAttentionBlock, self).__init__()
        self.in_features = in_features
        self.attn_features = attn_features
        self.subsample = subsample
        self.g = nn.Conv2d(in_channels=in_features, out_channels=attn_features, kernel_size=1, padding=0, bias=True)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=attn_features, out_channels=in_features, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_features, affine=True, track_running_stats=True)
        )
        if subsample:
            self.g = nn.Sequential(
                self.g,
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.phi = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        N, __, Wx, Hx = x.size()
        g_x = self.g(x).view(N,self.attn_features,-1).permute(0,2,1) # N x W'H' x C'
        theta_x = x.view(N,self.in_features,-1).permute(0,2,1) # N x WH x C
        if self.subsample:
            phi_x = self.phi(x).view(N,self.in_features,-1) # N x C x W'H'
        else:
            phi_x = x.view(N,self.in_features,-1) # N x C x WH
        c = torch.bmm(theta_x, phi_x) # N x WH x W'H'
        y = torch.bmm(F.softmax(c,dim=-1), g_x) # N x WH x C'
        y = y.permute(0,2,1).contiguous().view(N,self.attn_features,Wx,Hx)
        z = self.W(y)
        return z + x

'''
CBAM attention

Reference Paper:
Convolutional Block Attention Module https://arxiv.org/pdf/1807.06521.pdf

Reference code:
https://github.com/kobiso/CBAM-keras
'''
class CBAMAttentionBlock(nn.Module):
    # Convolutional Block Attention Module https://arxiv.org/abs/1807.06521
    def __init__(self, in_features, reduction=16, normalize_attn=False, reweight=False):
        super(CBAMAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.reweight = reweight
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1,1))
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=in_features//reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_features//reduction, out_channels=in_features, kernel_size=1, padding=0, bias=True)
        )
        # spatial attention
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
    def forward(self, x):
        N, C, __, __ = x.size()
        # channel attention
        avg_pool = self.mlp(self.avg_pool(x)) # N x C x 1 x 1
        max_pool = self.mlp(self.max_pool(x)) # N x C x 1 x 1
        if self.normalize_attn:
            channel_attn = F.softmax(avg_pool+max_pool, dim=1)
        else:
            channel_attn = torch.sigmoid(avg_pool+max_pool)
        x_channel_attn = x.mul(channel_attn)
        # spatial attention
        ch_avg_pool = torch.mean(x_channel_attn, dim=1, keepdim=True)
        ch_max_pool, __ = torch.max(x_channel_attn, dim=1, keepdim=True)
        c = self.conv(torch.cat((ch_avg_pool,ch_max_pool), 1))
        if self.normalize_attn:
            N, C, W, H = c.size() # C = 1
            spatial_attn = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
        else:
            spatial_attn = torch.sigmoid(c)
        output = x_channel_attn.mul(spatial_attn)
        if self.reweight:
            if self.normalize_attn:
                output = output.view(N,C,-1).sum(dim=2)
            else:
                output = F.adaptive_avg_pool2d(output, (1,1)).view(N,C)
        return c, output

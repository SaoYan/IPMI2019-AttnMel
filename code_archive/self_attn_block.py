import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_features, attn_features, subsample=True, mode='embedded_gaussian'):
        super(SelfAttentionBlock, self).__init__()
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product']
        self.mode = mode
        self.in_features = in_features
        self.attn_features = attn_features
        self.subsample = subsample
        self.g = nn.Conv2d(in_channels=in_features, out_channels=attn_features, kernel_size=1, padding=0, bias=True)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=attn_features, out_channels=in_features, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(in_features, affine=True, track_running_stats=True)
        )
        self.phi = None
        if mode in ['embedded_gaussian', 'dot_product']:
            self.theta = nn.Conv2d(in_channels=in_features, out_channels=attn_features, kernel_size=1, padding=0, bias=True)
            self.phi   = nn.Conv2d(in_channels=in_features, out_channels=attn_features, kernel_size=1, padding=0, bias=True)
        if subsample:
            self.g = nn.Sequential(
                self.g,
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            if self.phi is None:
                self.phi = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                self.phi = nn.Sequential(
                    self.phi,
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            return self._embedded_gaussian(x)
        elif self.mode == 'gaussian':
            return self._gaussian(x)
        elif self.mode == 'dot_product':
            return self._dot_product(x)
        else:
            raise NotImplementedError("Invalid non-local mode!")
    def _embedded_gaussian(self, x):
        N, __, Wx, Hx = x.size()
        g_x = self.g(x).view(N,self.attn_features,-1).permute(0,2,1) # N x W'H' x C'
        theta_x = self.theta(x).view(N,self.attn_features,-1).permute(0,2,1) # N x WH x C'
        phi_x = self.phi(x).view(N,self.attn_features,-1) # N x C' x W'H'
        c = torch.bmm(theta_x, phi_x) # N x WH x W'H'
        # __, Wc, Hc = c.size()
        y = torch.bmm(F.softmax(c,dim=-1), g_x) # N x WH x C'
        y = y.permute(0,2,1).contiguous().view(N,self.attn_features,Wx,Hx)
        z = self.W(y)
        return z + x
    def _gaussian(self, x):
        N, __, Wx, Hx = x.size()
        g_x = self.g(x).view(N,self.attn_features,-1).permute(0,2,1) # N x W'H' x C'
        theta_x = x.view(N,self.in_features,-1).permute(0,2,1) # N x WH x C
        if self.subsample:
            phi_x = self.phi(x).view(N,self.in_features,-1) # N x C x W'H'
        else:
            phi_x = x.view(N,self.in_features,-1) # N x C x WH
        c = torch.bmm(theta_x, phi_x) # N x WH x W'H'
        # __, Wc, Hc = c.size()
        y = torch.bmm(F.softmax(c,dim=-1), g_x) # N x WH x C'
        y = y.permute(0,2,1).contiguous().view(N,self.attn_features,Wx,Hx)
        z = self.W(y)
        return z + x
    def _dot_product(self, x):
        N, __, Wx, Hx = x.size()
        g_x = self.g(x).view(N,self.attn_features,-1).permute(0,2,1) # N x W'H' x C'
        theta_x = self.theta(x).view(N,self.attn_features,-1).permute(0,2,1) # N x WH x C'
        phi_x = self.phi(x).view(N,self.attn_features,-1) # N x C' x W'H'
        c = torch.bmm(theta_x, phi_x) # N x WH x W'H'
        __, Wc, Hc = c.size() # Hc = W'H'
        y = torch.bmm(c/Hc, g_x) # N x WH x C'
        y = y.permute(0,2,1).contiguous().view(N,self.attn_features,Wx,Hx)
        z = self.W(y)
        return z + x

'''
Channel-wise self attention
'''
class SelfAttentionBlock_chl(nn.Module):
    def __init__(self, in_features, attn_features, subsample=True):
        super(SelfAttentionBlock_chl, self).__init__()
        self.in_features = in_features
        self.attn_features = attn_features
        self.subsample = subsample
        self.phi = None
        if subsample:
            self.phi = nn.Conv2d(in_channels=in_features, out_channels=attn_features, kernel_size=1, padding=0, bias=True)
            self.g = nn.Conv2d(in_channels=in_features, out_channels=attn_features, kernel_size=1, padding=0, bias=True)
        self.theta = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.phi is not None:
            self.phi = nn.Sequential(
                self.phi,
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.phi = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        N, __, Wx, Hx = x.size()
        if self.subsample:
            g_x = self.g(x).view(N,self.attn_features,-1) # N x C' x WH
        else:
            g_x = x.view(N,self.in_features,-1) # N x C x WH
        theta_x = self.theta(x).view(N,self.in_features,-1) # N x C x W'H'
        if self.subsample:
            phi_x = self.phi(x).view(N,self.attn_features,-1).permute(0,2,1) # N x W'H' x C'
        else:
            phi_x = self.phi(x).view(N,self.in_features,-1).permute(0,2,1) # N x W'H' x C
        c = torch.bmm(theta_x, phi_x) # N x C x C or N x C x C'
        y = torch.bmm(F.softmax(c,dim=-1), g_x) # N x C x WH
        z = y.contiguous().view(N,self.in_features,Wx,Hx)
        return z + x

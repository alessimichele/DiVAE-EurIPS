import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

#=======================================================================================================================
# taken from: https://github.com/jmtomczak/vae_vampprior/tree/master/utils
#=======================================================================================================================
# WEIGHTS INITS
#=======================================================================================================================
def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def normal_init(m, mean: float = 0., std: float = 0.01):
    m.weight.data.normal_(mean, std)

#=======================================================================================================================

#=======================================================================================================================
# ACTIVATIONS
#=======================================================================================================================
class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat( F.relu(x), F.relu(-x), 1 )

#=======================================================================================================================
# LAYERS
#=======================================================================================================================
class NonLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, activation: nn.Module=None):
        super().__init__()

        self.activation = activation
        self.linear = nn.Linear(int(in_features), int(out_features), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)

        return h

#=======================================================================================================================
class GatedDense(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module=None):
        super().__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(in_features, out_features)
        self.g = nn.Linear(in_features, out_features)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation(h)

        g = self.sigmoid(self.g(x))

        return h * g
    
    @property
    def in_features(self):
        return self.h.in_features
    
    @property
    def out_features(self):
        return self.h.out_features

#=======================================================================================================================
class GatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int=1, activation: nn.Module=None):
        super().__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

#=======================================================================================================================
class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int=1, activation: nn.Module=None, bias: bool=True):
        super().__init__()

        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, x):
        h = self.conv(x)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out

# =======================================================================================================================
class Conv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(Conv2dBN, self).__init__()

        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out

#=======================================================================================================================
class ResUnitBN(nn.Module):
    '''
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Identity Mappings in Deep Residual Networks", https://arxiv.org/abs/1603.05027
    The unit used here is called the full pre-activation.
    '''
    def __init__(self, number_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ReLU(True)):
        super(ResUnitBN, self).__init__()

        self.activation = activation
        self.bn1 = nn.BatchNorm2d(number_channels)

        self.conv1 = nn.Conv2d(number_channels, number_channels, kernel_size, stride, padding, dilation)

        self.bn2 = nn.BatchNorm2d(number_channels)
        self.conv2 = nn.Conv2d(number_channels, number_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        residual = x

        h_bn_1 = self.bn1(x)
        h_act_bn_1 = self.activation(h_bn_1)
        h_1 = self.conv1(h_act_bn_1)

        h_bn_2 = self.bn2(h_1)
        h_act_bn_2 = self.activation(h_bn_2)
        h_2 = self.conv2(h_act_bn_2)

        out = h_2 + residual

        return out

# =======================================================================================================================
class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, scale_factor=2, activation=None):
        super(ResizeConv2d, self).__init__()

        self.activation = activation
        self.upsamplingNN = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        h = self.upsamplingNN(x)
        h = self.conv(h)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out

# =======================================================================================================================
class ResizeConv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, scale_factor=2,
                 activation=None):
        super(ResizeConv2dBN, self).__init__()

        self.activation = activation
        self.upsamplingNN = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = self.upsamplingNN(x)
        h = self.conv(h)
        h = self.bn(h)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out

# =======================================================================================================================
class ResizeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, scale_factor=2,
                 activation=None):
        super(ResizeGatedConv2d, self).__init__()

        self.activation = activation
        self.upsamplingNN = nn.Upsample(scale_factor=scale_factor)
        self.conv = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, activation=activation)

    def forward(self, x):
        h = self.upsamplingNN(x)
        out = self.conv(h)

        return out

# =======================================================================================================================
class GatedConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, dilation=1, activation=None):
        super(GatedConvTranspose2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation=dilation)
        self.g = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g

# =======================================================================================================================
class GatedResUnit(nn.Module):
    def __init__(self, in_channels, activation=None):
        super(GatedResUnit, self).__init__()

        self.activation = activation
        self.conv1 = GatedConv2d(in_channels, in_channels, 3, 1, 1, 1, activation=activation)
        self.conv2 = GatedConv2d(in_channels, in_channels, 3, 1, 1, 1, activation=activation)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)

        return h2 + x

#=======================================================================================================================
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

#=======================================================================================================================
class MaskedGatedConv2d(nn.Module):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedGatedConv2d, self).__init__()

        self.h = MaskedConv2d(mask_type, *args, **kwargs)
        self.g = MaskedConv2d(mask_type, *args, **kwargs)

    def forward(self, x):
        h = self.h(x)
        g = nn.Sigmoid()(self.g(x))
        return h * g

#=======================================================================================================================
class MaskedResUnit(nn.Module):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedResUnit, self).__init__()

        self.act = nn.ReLU(True)

        self.h1 = MaskedConv2d(mask_type, *args, **kwargs)
        self.h2 = MaskedConv2d(mask_type, *args, **kwargs)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        h1 = self.bn1(x)
        h1 = self.act(h1)
        h1 = self.h1(h1)

        h2 = self.bn2(h1)
        h2 = self.act(h2)
        h2 = self.h2(h2)
        return x + h2
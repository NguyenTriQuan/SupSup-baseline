import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
import math
from args import args
from .builder import Builder
import numpy as np

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, cfg, batch_norm=False):
        super(VGG, self).__init__()

        builder = Builder()
        size = args.input_size
        self.layers = make_layers(cfg, 3, batch_norm=batch_norm)

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.last_layers = nn.ModuleList([
            # nn.Dropout(),
            builder.conv1x1(512*self.smid*self.smid, 4096),
            nn.ReLU(True),
            # nn.Dropout(),
            builder.conv1x1(4096, 4096),
            nn.ReLU(True),
            builder.conv1x1(4096, args.output_size, last_layer=True)
        ])

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        for m in self.layers:
            x = m(x)

        x = x.view(x.size(0), -1, 1, 1)

        for m in self.last_layers:
            x = m(x)
        return x.squeeze()


def make_layers(cfg, n_channels, batch_norm=False):
    layers = []
    in_channels = n_channels
    builder = Builder()
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = builder.conv3x3(in_channels, v, stride=1)
            if batch_norm:
                layers += [conv2d, builder.batchnorm(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(cfg['A'], batch_norm=False)


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(cfg['A'], batch_norm=True)


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(cfg['B'], batch_norm=False)


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(cfg['B'], batch_norm=True)


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(cfg['C'], batch_norm=False)


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg['C'], batch_norm=True)


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(cfg['D'], batch_norm=False)


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(cfg['D'], batch_norm=True)

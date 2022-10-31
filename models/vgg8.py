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

class vgg8(nn.Module):
    def __init__(self):
        super().__init__()
        builder = Builder()
        size = 32
        self.conv1 = builder.conv3x3(3, 32, stride=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.conv2 = builder.conv3x3(32, 32, stride=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = builder.conv3x3(32, 64, stride=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.conv4 = builder.conv3x3(64, 64, stride=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = builder.conv3x3(64, 128, stride=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.conv6 = builder.conv3x3(128, 128, stride=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
#         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
#         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.smid = s
        self.fc1 = builder.conv1x1(s*s*128, 256)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=builder.conv1x1(256, args.output_size, last_layer=True)
        
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        h=self.relu(self.conv1(x))
        h=self.relu(self.conv2(h))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv3(h))
        h=self.relu(self.conv4(h))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv5(h))
        h=self.relu(self.conv6(h))
#         h=self.relu(self.conv7(h))
        h=self.drop1(self.MaxPool(h))
        h = h.view(h.size(0), -1, 1, 1)
        # h=h.view(x.shape[0],-1)
        h = self.drop2(self.relu(self.fc1(h)))
        
        return self.last(h).squeeze()
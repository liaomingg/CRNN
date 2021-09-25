# -*- coding: utf-8 -*-
# vgg.py
# author: lm

import torch 
import torch.nn as nn 


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 use_bn = False,
                 act='relu'):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2,
                              stride = 1,
                              bias = not use_bn)
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(True) if act=='relu' else nn.LeakyReLU(0.2, True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        
        return x 
    

class VGG7(nn.Module):
    def __init__(self,
                 in_channels = 1,
                 act='relu'
                 ):
        super(VGG7, self).__init__()
        assert act in ['relu', 'leaky_relu'], '{} is not supported.'
        self.conv0 = ConvBNReLU(in_channels, 64, kernel_size=3, act=act)
        self.pool0 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv1 = ConvBNReLU(64, 128, kernel_size=3, act=act)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = ConvBNReLU(128, 256, kernel_size=3, use_bn=True, act=act)
        
        self.conv3 = ConvBNReLU(256, 256, kernel_size=3, act=act)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        
        self.conv4 = ConvBNReLU(256, 512, kernel_size=3, use_bn=True, act=act)
        
        self.conv5 = ConvBNReLU(512, 512, kernel_size=3, act=act)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        
        self.conv6 = ConvBNReLU(512, 512, kernel_size=2, use_bn=True, act=act)
        self.out_channels = 512
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.pool0(x)
        
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        
        x = self.conv5(x)
        x = self.pool5(x)
        
        x = self.conv6(x)
        
        return x 
    
    
if __name__ == "__main__":
    from torchsummary import summary
    vgg7 = VGG7(1)
    print(vgg7)
    summary(vgg7, (1, 32, 280))
    torch.save(vgg7.state_dict(), 'vgg7.pth')
    
    
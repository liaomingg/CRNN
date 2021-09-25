# -*- coding: utf-8 -*-
# crnn.py
# author: lm

import torch
import torch.nn as nn
from .backbone.vgg7 import VGG7
from .head.bidirectional_head import CRNNHead
    

class CRNN(nn.Module):
    def __init__(self,
                 in_channels = 1,
                 hidden_features=256,
                 out_features=6228,
                 act='relu'
                 ):
        super(CRNN, self).__init__()
        assert act in ['relu', 'leaky_relu'], '{} is not supported.'
        self.backbone = VGG7(in_channels, act)
        self.head = CRNNHead(self.backbone.out_channels, hidden_features, out_features)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        _, _, h, _ = x.size()
        assert h == 1, 'Height must be 1.'
        x = x.squeeze(2) # [b, c, w]
        x = x.permute(2, 0, 1) # [b, c, w] -> [w/t, b, c]
        x = self.head(x) # [t, b, c]
        x = x.permute(1, 0, 2) # [b, t, c]
        
        return x 
    

if __name__ == "__main__":
    import torchsummary
    print('Test {}'.format(__file__))
    crnn = CRNN(1, out_features=6228)
    print(crnn)
    x = torch.randn(size=(1, 1, 32, 280))
    print('x.size():', x.size())
    y = crnn(x)
    print('y.size()', y.size())
    torch.save(crnn.state_dict(), 'crnn.pth')    
    
'''
CRNN(
  (backbone): VGG7(
    (conv0): ConvBNReLU(
      (conv): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (pool0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv1): ConvBNReLU(
      (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2): ConvBNReLU(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (conv3): ConvBNReLU(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv4): ConvBNReLU(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (conv5): ConvBNReLU(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (pool5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv6): ConvBNReLU(
      (conv): Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (head): CRNNHead(
    (rnn0): BidirectionalLSTM(
      (rnn): LSTM(512, 256, bidirectional=True)
      (emb): Linear(in_features=512, out_features=256, bias=True)
    )
    (rnn1): BidirectionalLSTM(
      (rnn): LSTM(256, 256, bidirectional=True)
      (emb): Linear(in_features=512, out_features=6228, bias=True)
    )
  )
)
'''
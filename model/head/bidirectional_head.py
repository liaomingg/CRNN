# -*- coding: utf-8 -*-
# bidirectional_head.py
# author: lm

import torch 
import torch.nn as nn 


class BidirectionalLSTM(nn.Module):
    '''双向LSTM'''
    def __init__(self,
                 in_features : int,
                 hidden_features : int,
                 out_features : int):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(in_features, hidden_features, bidirectional=True)
        self.emb = nn.Linear(hidden_features*2, out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x: shape: [t, b, c]
        :return y: shape: [t, b, c]
        '''
        x, _ = self.rnn(x)
        t, b, c = x.size()
        x = x.view(t*b, c)
        
        x = self.emb(x) # [t*b, out_features]
        x = x.view(t, b, -1) # [t, b, out_features]
        
        return x 
    

class CRNNHead(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int):
        super(CRNNHead, self).__init__()
        self.rnn0 = BidirectionalLSTM(in_features, hidden_features, hidden_features)
        self.rnn1 = BidirectionalLSTM(hidden_features, hidden_features, out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x: shape: [t, b, c].
        :return y: shape: [t, b, c].
        '''
        x = self.rnn0(x)
        x = self.rnn1(x)
        
        return x 
    
    
if __name__ == "__main__":
    from torchsummary import summary 
    head = CRNNHead(512, 256, 6228)
    print(head)
    x = torch.randn(size=(1, 16, 512))
    y = head(x)
    torch.save(head.state_dict(), './crnn_head.pth')
    
    
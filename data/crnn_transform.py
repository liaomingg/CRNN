# -*- coding: utf-8 -*-
# crnn_transform.py
# author: lm

import random
from numpy.core.fromnumeric import std

import torch
from PIL import Image, ImageOps
from torchvision import transforms


class ResizePadding(object):
    def __init__(self,
                 h = 32,
                 w = 280,
                 color = 0) -> None:
        super(ResizePadding, self).__init__()
        self.size = (w, h)
        self.color = color 
        
    def __call__(self, im : Image.Image) -> Image.Image:
        # maybe resize first, then padding.
        color = self.color if self.color else random.randint(0, 255)
        centering = random.choice([(0.5, 0.5), (0, 0), (1, 1)])
        method = random.choice([Image.CUBIC, Image.LINEAR, Image.NEAREST])
        im = ImageOps.pad(im, self.size, method, color, centering)
        return im 
        

class CRNNCollateFN(object):
    def __init__(self,
                 h = 32,
                 w = 280,
                 color = None) -> None:
        super(CRNNCollateFN, self).__init__()
        self.h = h 
        self.w = w 
        self.resize_padding = ResizePadding(h, w, color)
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                  std=(0.5, 0.5, 0.5))])
        
    def __call__(self, batch):
        '''将数据进行对齐'''
        images, labels = zip(*batch)
        
        images = [self.normalize(self.resize_padding(im)) for im in images]
        images = torch.cat([im.unsqueeze(0) for im in images])
        
        return images, labels 
        
        
if __name__ == "__main__":
    print('Test {}'.format(__file__))
    resize_padding = ResizePadding(32, 280, None)
    im = Image.open('imgs_words_en/word_545.png').convert('L')
    im = resize_padding(im)
    im.save('./resize_pad.jpg')

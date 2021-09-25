# -*- coding: utf-8 -*-
# label_converter.py
# author: lm

import torch
import os 
from collections import Iterable 

class CTCLabelConverter(object):
    def __init__(self,
                 alphabet,
                 ignore_case=False) -> None:
        super(CTCLabelConverter).__init__()
        self.ignore_case = ignore_case
        if alphabet.endswith('.py'):
            # load alphabet from .py.
            self.alphabet = self.load_py_alphabet(alphabet)
        elif alphabet.endswith('.txt'):
            # load alphabet from txt file.
            self.alphabet = self.load_txt_alphabet(alphabet)
        else:
            # write alphabet in alphabet
            self.alphabet = alphabet 
        
        self.dict = {}
        for i, char in enumerate(self.alphabet):
            # NOTE: index 0 is reserved for 'blank' required by CTC.
            self.dict[char] = i + 1
    
    def load_py_alphabet(self, path):
        from importlib import import_module
        module_path = os.path.splitext(path)[0].replace('/', '.')
        alphabet = import_module(module_path).alphabet
        return alphabet
        
    def load_txt_alphabet(self, path):
        '''将alphabet按顺序写在txt文件中'''
        return ''.join(open(path).read().split('\n'))
    
    def encode_one(self, text):
        '''注意label的长度与文本的长度有关'''
        label = [self.dict.get(char.lower() if self.ignore_case else char, 0) for char in text]
        length = [len(text)]
        return torch.IntTensor(label), torch.IntTensor(length)
    
    def encode_batch(self, texts):
        if isinstance(texts, str):
            label, length = self.encode_one(texts)
            
        elif isinstance(texts, Iterable):
            length = [len(text) for text in texts]
            text = ''.join(texts)
            label, _ = self.encode_one(text)
        return torch.IntTensor(label), torch.IntTensor(length)
    
    
    def decode_one(self, label : torch.Tensor, length, raw=False):
        '''Decode one label to string.'''
        assert label.numel() == length, 'label with length: {} does not match declared length: {}.'.format(label.numel(), length)
        if raw:
            return ''.join([self.alphabet[i-1] for i in label])
        else:
            chars = []
            for i in range(length):
                if label[i] != 0 and (not (i > 0 and label[i-1] == label[i])):
                    chars.append(self.alphabet[label[i]-1])
            return ''.join(chars)
    
    def decode_batch(self, label : torch.Tensor, length : torch.Tensor, raw=False):
        '''Decode enocded texts back to strs.'''
        assert label.numel() == length.sum(), 'labels with length: {} does not match with declared length: {}'.format(label.numel(), length.sum())
        texts = []
        start = 0
        for i in range(length.numel()):
            texts.append(self.decode_one(label[start:start+length[i]], length[i], raw))
            start += length[i]
        return texts 
                

if __name__ == "__main__":
    import string 
    print('Test {}'.format(__file__))
    convert = CTCLabelConverter(alphabet='../keys/keys_demo.py')
    label, length = convert.encode_batch(['liaoming', 'hello'])
    print(label, length)
    texts = convert.decode_batch(torch.tensor(label, dtype=torch.int32), torch.tensor(length, dtype=torch.int32), raw=True)
    print(texts)
    
    
    
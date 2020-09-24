import torch
import torch.nn as nn
from collections import OrderedDict

"""
Model definition for a LeNet-5
http://yann.lecun.com/exdb/lenet/
"""
class C1(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1',nn.Conv2d(1,20, kernel_size=5)),
            ('ramp1',nn.ReLU()),
            ('pool1',nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
    
    def forward(self,img):
        output = self.c1(img)
        return output

class C2(nn.Module):
    def __init__(self):
        super().__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2',nn.Conv2d(20,50, kernel_size=5)),
            ('ramp2',nn.ReLU()),
            ('pool2',nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
    
    def forward(self,img):
        output = self.c2(img)
        return output 

class L1(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Sequential(OrderedDict([
            ('l1',nn.Linear(1250,500)),
            ('ramp3',nn.ReLU())
        ]))
    
    def forward(self,img):
        output = self.l1(img)
        return output 

class L2(nn.Module):
    def __init__(self):
        super().__init__()

        self.l2 = nn.Sequential(OrderedDict([
            ('l2',nn.Linear(500,10)),
            ('softmax',nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self,img):
        output = self.l2(img)
        return output

class LeNetNN(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super().__init__()

        self.c1 = C1()
        self.c2 = C2() 
        self.l1 = L1() 
        self.l2 = L2() 

    def forward(self, img):
        out = self.c1(img)
        out = self.c2(out)
        out = torch.flatten(out, 1)
        out = self.l1(out)
        out = self.l2(out)
        return out


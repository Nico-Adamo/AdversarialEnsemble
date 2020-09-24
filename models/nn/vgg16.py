import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Model definition for VGG-16 (easily modifiable to any VGG)
https://arxiv.org/abs/1409.1556
"""
class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True))
    
    def forward(self,img):
        output = self.conv(img)
        return output 

class VGG16NN(nn.Module):
    def build_layers(self,cfg):
        in_channels = 3 # CIFAR-10
        layers = []
        for layer in cfg:
            if layer=="P":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [ConvBlock(in_channels, layer)]
                in_channels = layer
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def __init__(self):
        super().__init__()

        # For lower memory (<8GB) GPUs 
        # self.cfg = [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P']
        
        self.cfg = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']
        self.network = self.build_layers(self.cfg)
        self.classifier = nn.Linear(512, 10)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.network(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = self.softmax(out)
        return out

    
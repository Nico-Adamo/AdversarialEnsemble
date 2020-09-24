import torch
import torch.nn as nn 
from models.nn.lenet import LeNetNN
from models.nn.vgg16 import VGG16NN
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from models.ensemble_model import EnsembleModel


"""
Ensemble networks - derived from EnsembleModel (ensemble_model.py)
Defining DataLoaders, loss function, instantiating nets, etc. 
"""

DOWNLOAD_DATASETS = False # Set to true if downloading mnist and/or cifar for the first time. 
                          # Warning: Will spam your terminal. 

class LeNetEnsemble(EnsembleModel):
    name = "LeNet-5"
    classes = ["0","1","2","3","4","5","6","7","8","9"]
    data_train = MNIST('./data/mnist',
                        download=DOWNLOAD_DATASETS,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor()]))
    data_test = MNIST('./data/mnist',
                  train=False,
                  download=DOWNLOAD_DATASETS,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

    data_train_loader = DataLoader(
        data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(
        data_test, batch_size=1024, num_workers=8)

    loss_criterion = nn.CrossEntropyLoss()
    
    def __init__(self, num_nets):
        super().__init__(num_nets)
        self.nets = [LeNetNN() for _ in range(num_nets)]
        self.optimizers = [optim.Adam(net.parameters(), lr=2e-3) for net in self.nets]


class VGGEnsemble(EnsembleModel):
    name = "VGG-16"
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    data_train = CIFAR10(
        root='./data', train=True, download=DOWNLOAD_DATASETS,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    data_test = CIFAR10(
        root='./data', train=DOWNLOAD_DATASETS, download=False, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    
    data_train_loader = DataLoader(
            data_train, batch_size=64, shuffle=True, num_workers=2)
    data_test_loader = DataLoader(
        data_test, batch_size=100, shuffle=False, num_workers=2)

    loss_criterion = nn.CrossEntropyLoss()

    def __init__(self, num_nets):
        super().__init__(num_nets)
        self.nets = [VGG16NN() for _ in range(num_nets)]
        self.optimizers = [optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) for net in self.nets]
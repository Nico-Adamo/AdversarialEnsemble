from abc import ABC, abstractmethod
import torch

"""
Abstract Base Class representing the structure of an ensemble model,
including class properties and abstract methods.  
"""
class EnsembleModel(ABC):
    def __init__(self,num_nets):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    @property # Name of model - String
    def name(self): 
        raise NotImplementedError

    @property # String names corresponding to classifier labels - List of Strings
    def classes(self): 
        raise NotImplementedError

    @property # Training data including transformations - torch.utils.data.Dataset
    def data_train(self): 
        raise NotImplementedError
    
    @property # Test data including transformations - torch.utils.data.Dataset
    def data_test(self):
        raise NotImplementedError

    @property # Training dataloader - torch.utils.data.DataLoader
    def data_train_loader(self):
        raise NotImplementedError

    @property # Test dataloader - torch.utils.data.DataLoader
    def data_test_loader(self):
        raise NotImplementedError

    @property # Loss function - i.e torch.nn.CrossEntropyLoss, torch.nn.NLLLoss()
    def loss_criterion(self):
        raise NotImplementedError

    @abstractmethod # Training loop for one net, for one epoch
    def train_net_epoch(self, net, optimizer, epoch, print_batch=False):
        pass

    # Training loop for one net, for one epoch
    def train_net_epoch(self, net, optimizer, epoch, print_batch=False):
        net.to(self.device)
        net.train()
        loss_list, batch_list = [], []
        for i, (images, labels) in enumerate(self.data_train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()

            output = net(images)

            loss = self.loss_criterion(output, labels)

            loss_list.append(loss.detach().cpu().item())
            batch_list.append(i+1)

            if print_batch==True and i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

            loss.backward()
            optimizer.step()

    # Calculate test loss for one net
    def test_loss(self,net):
        net.eval()
        total_correct = 0
        avg_loss = 0.0
        for i, (images, labels) in enumerate(self.data_test_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            output = net(images)
            avg_loss += self.loss_criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

        avg_loss /= len(self.data_test)
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(self.data_test)))
        return avg_loss

    # Train all nets for a certain number of epochs
    def train_nets(self,epochs,print_batch=False):
        for netIndex in range(len(self.nets)):
            print("Training Net " + str(netIndex) + "...")
            for epoch in range(epochs):
                self.train_net_epoch(self.nets[netIndex],self.optimizers[netIndex],epoch,print_batch=print_batch)
            self.test_loss(self.nets[netIndex])
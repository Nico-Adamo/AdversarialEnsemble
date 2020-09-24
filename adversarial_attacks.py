import torch
import torch.nn as nn
import torch.nn.functional as F

def input_gradient(net,image,target,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    image, target = image.to(device), target.to(device)
    image.requires_grad = True
    net.zero_grad()
    output = net(image)
    loss = F.nll_loss(output, target)
    loss.backward()
    return image.grad.data


def targeted_fgsm(net,image,target,iterations=15,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),epsilon=0.01):
    image, target = image.to(device), target.to(device)
    for i in range(iterations):
        data_grad = input_gradient(net,image,target,device=device)
        sign_data_grad = data_grad.sign()
        image = image.detach_() - epsilon*sign_data_grad
        image = torch.clamp(image, 0, 1)
    return image

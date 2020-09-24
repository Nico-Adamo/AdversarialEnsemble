import torch
from models.model_definitions import LeNetEnsemble
from models.model_definitions import VGGEnsemble
from collections import OrderedDict
from adversarial_attacks import targeted_fgsm
from torchvision.utils import make_grid
from torchvision.utils import save_image

"""
Helper functions and pretty printing for generating batches of 
adversarial examples and measuring their transfer over ensemble networks
"""

"""Experiment Parameters"""
NUM_ADVERSARIAL_EXAMPLES = 100
ENSEMBLE_SIZE = 5
ENSEMBLE_MODEL_LIST = [LeNetEnsemble]

def generate_adversarial_examples(net, init_tensor, target, num_examples, iterations=15):
    try:
        target_array = torch.full(torch.Size([num_examples]), target)
    except RuntimeError:
        target_array = torch.full(torch.Size([num_examples]), target, dtype=torch.long)

    return targeted_fgsm(net,
                init_tensor,
                target_array,
                iterations=iterations)

def count_adversarial_labels(net, adv_examples):
    adv_labels = net(adv_examples).max(1)[1].tolist()
    label_freq = {}
    for label in adv_labels: 
        if (label in label_freq): 
            label_freq[label] += 1
        else: 
            label_freq[label] = 1
    return label_freq

def measure_adversarial_transfer(model,target,train_epochs=1,save_images=False):
    model.train_nets(train_epochs,print_batch=False)
    adv_net = model.nets[0]
    adversarial_input_list = generate_adversarial_examples(
                adv_net, 
                torch.rand([NUM_ADVERSARIAL_EXAMPLES,1,32,32]), 
                target, 
                NUM_ADVERSARIAL_EXAMPLES, 
                iterations=100)

    if save_images:
        save_image(make_grid(adversarial_input_list), model.name+"_adversarial_"+str(model.classes[target])+".png",nrow=10)

    print("\n-- " + model.name + " --")
    print("Adversarial Label: "+str(model.classes[target]))
    for net_index in range(len(model.nets)):
        print("### Network "+ str(net_index) + " ###")
        for label, count in count_adversarial_labels(adv_net, adversarial_input_list).items(): 
            print ("Label % s:% d%%"%(model.classes[label], count/NUM_ADVERSARIAL_EXAMPLES*100))
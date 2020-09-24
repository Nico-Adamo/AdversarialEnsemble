# AdversarialEnsemble
Measuring the transfer of adversarial examples over ensemble networks of the same architecture (VGG-16 and LeNet-5)
Testing the theory of whether adversarial examples take on arbitrary labels under the same models trained w/ different initializations.

`models` contains an abstract base class implementation of an ensemble network, as well as the specific instantiations for VGG-16 on Cifar and LeNet on MNIST. 
`models/nn` contains an implementation of LeNet-5 and VGG-16 - shouldn't be too hard to add other models. 

Re-implementation from Mathematica, found here:
https://github.com/Nico-Adamo/WSC2019-Colorful-Fraud

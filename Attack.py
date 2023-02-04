import torch
import torch.nn as nn
import torch.optim as optim

def fgsm(model, inputs,epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(inputs, requires_grad=True)
    
    scores = torch.sigmoid(model(inputs+delta)).squeeze()

    scores.backward()

    
    return epsilon * delta.grad.detach().sign()


def pgd(model, inputs, epsilon, alpha, num_iter):

    delta = torch.zeros_like(inputs, requires_grad=True)
    for t in range(num_iter):
        

        scores = torch.sigmoid(model(inputs+delta)).squeeze()  
        scores.backward()
        
        delta.data = (delta + inputs.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()
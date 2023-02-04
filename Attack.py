import torch
import torch.nn as nn
import torch.optim as optim

def fgsm(model, inputs,epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(inputs, requires_grad=True)
    
    scores = torch.sigmoid(model(inputs+delta)).squeeze()

    scores.backward()

    # return inputs+epsilon * delta.grad.detach().sign()
    return epsilon * delta.grad.detach().sign()


def pgd(model, inputs, c, epsilon, alpha, num_iter,objective,R):

    delta = torch.zeros_like(inputs, requires_grad=True)
    for t in range(num_iter):
        

        outputs = model(inputs+delta)
        dist = torch.sum((outputs - c) ** 2, dim=1)
        if objective == 'soft-boundary':
            scores = dist - R ** 2
        else:
            scores = dist
        scores.backward()        
        
        
        
        delta.data = (delta + inputs.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()
import torch
import torch.nn as nn

from attack_torch import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ['default', 'targeted']
        
        self.model=model

    def forward(self, images,labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        images.requires_grad = True
        
        cost=getScore(self.model,images)

        
        
        
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        
        adv_images= images+self.eps*grad.sign() if labels==0 else images-self.eps*grad.sign()
        
        # adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        # print(f'Label :  {labels}')
        # print(f'new Score delta: {getScore(self.model,x)}')
        # print(f'new Score delta: {getScore(self.model,x)}\n\n')
        

        return adv_images
    
def getScore(f,x):
    scores = torch.sigmoid(f(x)).squeeze()
    return scores

    
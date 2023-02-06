import torch

from util.cifar import CIFAR10, CIFAR100OE
from util.cifar100 import *
from util.mnist import *
from util.fashionmnist import *
from util.svhn import *
from util.mvtec import *


def cifar10(config):
    assert config.normal_class in range(10), "Set normal_class to 0-9."

    cifar10 = CIFAR10(root=config.data_path,
        normal_class=config.normal_class,
        hold_one_out=config.benchmark == "hold_one_out")
    cifar100 = CIFAR100OE(root=config.data_path)

    train_loader = torch.utils.data.DataLoader(dataset=cifar10.train_set,
        batch_size=config.batch_size//2,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=cifar10.test_set,
        batch_size=1,
        num_workers=1,
        pin_memory=True)
    
    oe_loader = torch.utils.data.DataLoader(dataset=cifar100.oe_set,
        batch_size=config.batch_size//2,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=True)

    return train_loader, oe_loader, val_loader


def mnist(config):
    assert config.normal_class in range(10), "Set normal_class to 0-9."

    mnist = MNIST(root=config.data_path,
        normal_class=config.normal_class,
        hold_one_out=config.benchmark == "hold_one_out")
    
    cifar100 = MN_CIFAR100OE(root=config.data_path)
    

    train_loader = torch.utils.data.DataLoader(dataset=mnist.train_set,
        batch_size=config.batch_size//2,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=mnist.test_set,
        batch_size=1,
        num_workers=1,
        pin_memory=True)
    
    oe_loader = torch.utils.data.DataLoader(dataset=cifar100.oe_set,
        batch_size=config.batch_size//2,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    
    
    

    return train_loader, None, val_loader



def fashionmnist(config):
    assert config.normal_class in range(10), "Set normal_class to 0-9."

    fashionmnist = FashionMNIST(root=config.data_path,
        normal_class=config.normal_class,
        hold_one_out=config.benchmark == "hold_one_out")
    
    cifar100 = FM_CIFAR100OE(root=config.data_path)
    
    train_loader = torch.utils.data.DataLoader(dataset=fashionmnist.train_set,
        batch_size=config.batch_size//2,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=fashionmnist.test_set,
        batch_size=1,
        num_workers=1,
        pin_memory=True)
    
    oe_loader = torch.utils.data.DataLoader(dataset=cifar100.oe_set,
        batch_size=config.batch_size//2,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    


    return train_loader, None, val_loader


def svhn(config):
    assert config.normal_class in range(10), "Set normal_class to 0-9."

    svhn = SVHN(root=config.data_path,
        normal_class=config.normal_class,
        hold_one_out=config.benchmark == "hold_one_out")
    
    cifar100 = SV_CIFAR100OE(root=config.data_path)
    
    train_loader = torch.utils.data.DataLoader(dataset=svhn.train_set,
        batch_size=config.batch_size//2,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=svhn.test_set,
        batch_size=1,
        num_workers=1,
        pin_memory=True)
    
    oe_loader = torch.utils.data.DataLoader(dataset=cifar100.oe_set,
        batch_size=config.batch_size//2,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    


    return train_loader, None, val_loader

def cifar100(config):
    assert config.normal_class in range(10), "Set normal_class to 0-9."

    fashionmnist = CIFAR100(root=config.data_path,
        normal_class=config.normal_class,
        hold_one_out=config.benchmark == "hold_one_out")
    
    
    cifar100 = CIFAR10OE(root=config.data_path)

    train_loader = torch.utils.data.DataLoader(dataset=fashionmnist.train_set,
        batch_size=config.batch_size//2,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=fashionmnist.test_set,
        batch_size=1,
        num_workers=1,
        pin_memory=True)
    
    oe_loader = torch.utils.data.DataLoader(dataset=cifar100.oe_set,
        batch_size=config.batch_size//2,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    


    return train_loader, None, val_loader

def mvtec(config):
    assert config.normal_class in range(15), "Set normal_class to 0-9."

    fashionmnist = MVTec(root=config.data_path,
        normal_class=config.normal_class,
        hold_one_out=config.benchmark == "hold_one_out")
    


    cifar100 = MV_CIFAR100OE(root=config.data_path)
        
    train_loader = torch.utils.data.DataLoader(dataset=fashionmnist.train_set,
        batch_size=config.batch_size//2,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=fashionmnist.test_set,
        batch_size=1,
        num_workers=1,
        pin_memory=True)
    
    oe_loader = torch.utils.data.DataLoader(dataset=cifar100.oe_set,
        batch_size=config.batch_size//2,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=True)
    


    return train_loader, None, val_loader


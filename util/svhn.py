import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset


class SVHN(torch.utils.data.Dataset):
    def __init__(self, root, normal_class, hold_one_out=False, img_size=32):
        super().__init__()
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if hold_one_out:
            self.normal_classes, self.outlier_classes = self.outlier_classes, self.normal_classes

        train_transform = [
          transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4309803921568628, 0.4301960784313726, 0.4462745098039216), (0.19647058823529412, 0.1984313725490196, 0.19921568627450978))]

        test_transform = [
          transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4309803921568628, 0.4301960784313726, 0.4462745098039216), (0.19647058823529412, 0.1984313725490196, 0.19921568627450978))]

        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)
        
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        dataset = torchvision.datasets.SVHN(root=root,
            split='train',
            transform=train_transform,
            target_transform=target_transform,
            download=True)

        idx = np.argwhere(np.isin(np.array(dataset.labels), self.normal_classes))
        idx = idx.flatten().tolist()
        
        self.train_set = Subset(dataset, idx)
        self.test_set = torchvision.datasets.SVHN(root=root,
            split='test',
            transform=test_transform,
            target_transform=target_transform,
            download=True)





class SV_CIFAR100OE(torch.utils.data.Dataset):
    def __init__(self, root, img_size=32, n_classes=100):
        super().__init__()
        self.normal_classes = None
        self.outlier_classes = list(range(0, 100))
        self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_classes))
    
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491373, 0.482353, 0.446667), (0.247059, 0.243529, 0.261569))])

        dataset = torchvision.datasets.CIFAR100(root=root,
            train=True,
            transform=transform,
            target_transform=transforms.Lambda(lambda x: int(x in self.outlier_classes)),
            download=True)

        idx = np.argwhere(np.isin(np.array(dataset.targets), self.known_outlier_classes))
        idx = idx.flatten().tolist()

        # Filter classes utilized in outlier exposure
        self.oe_set = Subset(dataset, idx)
        self.oe_set.shuffle_idxs = False

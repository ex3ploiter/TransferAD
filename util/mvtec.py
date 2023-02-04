import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset

import os 
import glob
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToPILImage


class MVTec(torch.utils.data.Dataset):
    def __init__(self, root, normal_class, hold_one_out=False, img_size=256):
        super().__init__()
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 15))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if hold_one_out:
            self.normal_classes, self.outlier_classes = self.outlier_classes, self.normal_classes

        train_transform = [
          torchvision.transforms.Grayscale(num_output_channels=3),
          transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))]

        test_transform = [
          torchvision.transforms.Grayscale(num_output_channels=3),
          transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))]

        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)
        
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))


        self.train_set = torchvision.datasets.MyMVTec(root=root,
            train=True,
            transform=train_transform,
            target_transform=target_transform,
            normal_class= self.normal_classes)

        
        self.test_set = torchvision.datasets.MyMVTec(root=root,
            train=False,
            transform=train_transform,
            target_transform=target_transform,
            normal_class= self.normal_classes)





class MyMVTec(Dataset):
    def __init__(self, root, normal_class, transform=None, target_transform=None, train=True):
        self.transform = transform
        root=os.path.join(root,'mvtec_anomaly_detection')
        
        mvtec_labels=['bottle' , 'cable' , 'capsule' , 'carpet' ,'grid' , 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood','zipper']
        category=mvtec_labels[normal_class]

        if train:
            self.image_files = glob.glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
          image_files = glob.glob(os.path.join(root, category, "test", "*", "*.png"))
          normal_image_files = glob.glob(os.path.join(root, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          


          self.image_files = normal_image_files+anomaly_image_files

        
        

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1
        
        return image, target
        

    def __len__(self):
        return len(self.image_files)
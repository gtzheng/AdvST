from __future__ import print_function, absolute_import, division
import h5py
import os
import bz2
import scipy
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url
from common.utils import unfold_label, shuffle_data
from collections import Counter
from common.autoaugment import SVHNPolicy, CIFAR10Policy
from common.randaugment import RandAugment
from functools import partial
from torch.utils.data import Dataset
from torchvision import transforms
# Dataset information: http://sketchx.eecs.qmul.ac.uk/downloads/
# https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ

class Denormalise(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-12)
        mean_inv = -mean * std_inv
        super(Denormalise, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(Denormalise, self).__call__(tensor.clone())


class PACS(Dataset):
    def __init__(self, root_folder, name, split='train', transform=None, ratio=None):
        path = os.path.join(root_folder, '{}_{}.hdf5'.format(name,split))
        if split == 'train':
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transform
        else:
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transform

        f = h5py.File(path, "r")
        self.x = np.array(f['images'])
        self.y = np.array(f['labels'])
        self.op_labels = torch.tensor(np.ones(len(self.y),dtype=np.int)*(-1))
        if ratio is not None:
            num = len(self.x)
            indexes = np.random.permutation(num)
            sel_num = int(ratio * num)
            self.x = self.x[indexes[0:sel_num]]
            self.y = self.y[indexes[0:sel_num]]
            self.op_labels = self.op_labels[indexes[0:sel_num]]
        f.close()
        def resize(x):
            x = x[:, :,
                [2, 1, 0]]  # we use the pre-read hdf5 data file from the download page and need to change BGR to RGB
            x = x.astype(np.uint8)
            return np.array(Image.fromarray(obj=x, mode='RGB').resize(size=(224, 224)))
        self.x = np.array(list(map(resize, self.x)))
        self.x = torch.tensor(self.x).permute(0,3,1,2)
        self.y -= np.min(self.y)
        self.y = torch.tensor(self.y.astype(np.int64))
        self.preprocess = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        self.image_denormalise = Denormalise([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]
        if op < 0:
            x = transforms.ToPILImage()(x)
            x = self.transform(x)
        return x, y, op


class PACSMultiple(Dataset):
    def __init__(self, root_folder, names, split='train', transform=None):
        
        if split == 'train':
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4, .4, .4, .4),
                    # transforms.RandomCrop(224, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transform
        else:
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transform
        def resize(x):
            x = x[:, :,
                [2, 1, 0]]  # we use the pre-read hdf5 data file from the download page and need to change BGR to RGB
            x = x.astype(np.uint8)
            return np.array(Image.fromarray(obj=x, mode='RGB').resize(size=(224, 224)))
        self.x = []
        self.y = []
        for name in names:
            path = os.path.join(root_folder, '{}_{}.hdf5'.format(name,split))
            f = h5py.File(path, "r")
            x = np.array(f['images'])
            y = np.array(f['labels'])
            f.close()
            x = np.array(list(map(resize, x)))
            y -= np.min(y)
            y = y.astype(np.int64)

            self.x.append(x)
            self.y.append(y)

        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        x = transforms.ToPILImage()(x)
        x = self.transform(x)
        return x, y

class PACSTensor(Dataset):
    def __init__(self, root_folder, name, split='train', transform=None):
        path = os.path.join(root_folder, '{}_{}.hdf5'.format(name,split))
        
        f = h5py.File(path, "r")
        self.x = np.array(f['images'])
        self.y = np.array(f['labels'])
        f.close()
        def resize(x):
            x = x[:, :,
                [2, 1, 0]]  # we use the pre-read hdf5 data file from the download page and need to change BGR to RGB
            x = x.astype(np.uint8)
            return np.array(Image.fromarray(obj=x, mode='RGB').resize(size=(224, 224)))
        self.x = np.array(list(map(resize, self.x)))
        self.x = torch.tensor(self.normalize(self.x),dtype=torch.float32)
        self.y -= np.min(self.y)
        self.y = torch.tensor(self.y,dtype=torch.long)

    def __len__(self):
        return len(self.x)


    def normalize(self, inputs):

        # the mean and std used for the normalization of
        # the inputs for the pytorch pretrained model
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # norm to [0, 1]
        inputs = inputs / 255.0

        inputs_norm = []
        for item in inputs:
            item = np.transpose(item, (2, 0, 1))
            item_norm = []
            for c, m, s in zip(item, mean, std):
                c = np.subtract(c, m)
                c = np.divide(c, s)
                item_norm.append(c)

            item_norm = np.stack(item_norm)
            inputs_norm.append(item_norm)

        inputs_norm = np.stack(inputs_norm)

        return inputs_norm

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y



class PACSTensorMultiple(Dataset):
    def __init__(self, root_folder, names, split='train', transform=None):
        def resize(x):
            x = x[:, :,
                [2, 1, 0]]  # we use the pre-read hdf5 data file from the download page and need to change BGR to RGB
            x = x.astype(np.uint8)
            return np.array(Image.fromarray(obj=x, mode='RGB').resize(size=(224, 224)))
        self.x = []
        self.y = []
        for name in names:
            path = os.path.join(root_folder, '{}_{}.hdf5'.format(name,split))
            f = h5py.File(path, "r")
            x = np.array(f['images'])
            y = np.array(f['labels'])
            f.close()
            x = np.array(list(map(resize, x)))
            x = torch.tensor(self.normalize(x),dtype=torch.float32)
            y -= np.min(y)
            y = torch.tensor(y,dtype=torch.long)

            self.x.append(x)
            self.y.append(y)
        self.x = torch.cat(self.x)
        self.y = torch.cat(self.y)

    def __len__(self):
        return len(self.x)


    def normalize(self, inputs):

        # the mean and std used for the normalization of
        # the inputs for the pytorch pretrained model
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # norm to [0, 1]
        inputs = inputs / 255.0

        inputs_norm = []
        for item in inputs:
            item = np.transpose(item, (2, 0, 1))
            item_norm = []
            for c, m, s in zip(item, mean, std):
                c = np.subtract(c, m)
                c = np.divide(c, s)
                item_norm.append(c)

            item_norm = np.stack(item_norm)
            inputs_norm.append(item_norm)

        inputs_norm = np.stack(inputs_norm)

        return inputs_norm

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y



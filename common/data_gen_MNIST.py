from __future__ import print_function, absolute_import, division

import os
import bz2
import scipy
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
from torchvision.datasets import MNIST, SVHN
from torchvision.datasets.utils import download_url
from common.utils import unfold_label, shuffle_data
from collections import Counter
from common.autoaugment import SVHNPolicy, CIFAR10Policy
from common.randaugment import RandAugment
from functools import partial
from torch.utils.data import Dataset
_image_size = 32
_trans = transforms.Compose([
    transforms.Resize(_image_size),
    transforms.ToTensor()
])

class DigitsDataset(Dataset):
    def __init__(self, x, y, aug='', con=0):
        self.x = x
        self.y = y
        self.op_labels = torch.tensor(np.ones(len(self.y),dtype=np.int)*(-1))
        self.con = con
        if aug == '':
            self.transform = None
        else:
            transform = [transforms.ToPILImage()]
            if aug == 'AA':
                transform.append(SVHNPolicy())
            elif aug == 'RA':
                transform.append(RandAugment(3,4))

            transform.append(transforms.ToTensor())
            transform = transforms.Compose(transform)
            self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]
        if self.transform is not None:
            if self.con>0:
                aug_x = []
                for i in range(self.con):
                    aug_x.append(self.transform(x))
                aug_x = torch.stack(aug_x,0)
                aug_y = torch.stack([y,y])
                return aug_x, aug_y
            else:
                x = self.transform(x)
        return x, y, op

def get_data_loaders():
    return [
        ['MNIST', 'SVHN', 'MNIST_M', 'SYN', 'USPS'],
        [load_mnist, load_svhn, load_mnist_m, load_syn, load_usps]
    ]

def get_data_loaders_imbalanced(ratio):
    load_mnist_partial = partial(load_mnist_imbalanced, ratio=ratio)
    return [
        ['MNIST', 'SVHN', 'MNIST_M', 'SYN', 'USPS'],
        [load_mnist_partial, load_svhn, load_mnist_m, load_syn, load_usps]
    ]

def load_mnist_imbalanced(root_dir, train=True, ratio=2.0):
    # ratio = n_major / n_minor
    N = 10000
    major_classes = [0,1]
    minor_classes = [2,3,4,5,6,7,8,9]
    n_minor = int(N/(len(major_classes)*ratio+len(minor_classes)))
    n_major = int((N-len(minor_classes)*n_minor)/len(major_classes))
    print('Ratio: {:.4f}, n_major/n_minor={}/{}'.format(ratio,n_major,n_minor))
    dataset = MNIST(root_dir, train=train, download=True, transform=_trans)
    labels = []
    
    class_dict = {}
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.expand(3, -1, -1).numpy()
        if class_dict.get(label,None) is None:
            class_dict[label] = [image]
        else:
            class_dict[label].append(image)
        labels.append(label)
    labels = np.array(labels)
    statistics = Counter(labels)
    if n_major > np.array(list(statistics.values())).min():
        raise Exception("Not enough samples")
    images = []
    labels = []
    for c in major_classes:
        images.extend(class_dict[c][0:n_major])
        labels.extend([c]*n_major)
    for c in minor_classes:
        images.extend(class_dict[c][0:n_minor])
        labels.extend([c]*n_minor)
    images, labels = np.stack(images), np.array(labels)
    return images, labels

def load_mnist(root_dir, train=True, aug='', con=0):
    dataset = MNIST(root_dir, train=train, download=True, transform=_trans)
    images, labels = [], []

    for i in range(10000 if train else len(dataset)):
        image, label = dataset[i]
        images.append(image.expand(3, -1, -1).numpy())
        labels.append(label)
    images, labels = torch.tensor(np.stack(images)), torch.tensor(np.array(labels))
    dataset = DigitsDataset(images, labels, aug, con)
    return dataset

def load_svhn(root_dir, train=True, aug='', con=0):
    split = 'train' if train else 'test'
    dataset = SVHN(os.path.join(root_dir, 'SVHN'), split=split, download=True, transform=_trans)
    images, labels = [], []

    for i in range(len(dataset)):
        image, label = dataset[i]
        images.append(image.numpy())
        labels.append(label)
    images, labels = torch.tensor(np.stack(images)), torch.tensor(np.array(labels))
    dataset = DigitsDataset(images, labels, aug, con)
    return dataset


def load_usps(root_dir, train=True, aug='', con=0):
    split_list = {
        'train': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
            "usps.bz2", 'ec16c51db3855ca6c91edd34d0e9b197'
        ],
        'test': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
            "usps.t.bz2", '8ea070ee2aca1ac39742fdd1ef5ed118'
        ],
    }

    split = 'train' if train else 'test'
    url, filename, checksum = split_list[split]
    root = os.path.join(root_dir, 'USPS')
    full_path = os.path.join(root, filename)

    if not os.path.exists(full_path):
        download_url(url, root, filename, md5=checksum)

    with bz2.BZ2File(full_path) as fp:
        raw_data = [l.decode().split() for l in fp.readlines()]
        imgs = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
        imgs = np.asarray(imgs, dtype=np.float32).reshape((-1, 16, 16))
        imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
        targets = [int(d[0]) - 1 for d in raw_data]

    images, labels = [], []
    for img, target in zip(imgs, targets):
        img = Image.fromarray(img, mode='L')
        img = _trans(img)
        images.append(img.expand(3, -1, -1).numpy())
        labels.append(target)
    images, labels = torch.tensor(np.stack(images)), torch.tensor(np.array(labels))
    dataset = DigitsDataset(images, labels, aug, con)
    return dataset


def load_syn(root_dir, train=True, aug='', con=0):
    split_list = {
        'train': "synth_train_32x32.mat",
        'test': "synth_test_32x32.mat"
    }

    split = 'train' if train else 'test'
    filename = split_list[split]
    full_path = os.path.join(root_dir, 'SYN', filename)

    raw_data = scipy.io.loadmat(full_path)
    imgs = np.transpose(raw_data['X'], [3, 0, 1, 2])
    images = []
    for img in imgs:
        img = Image.fromarray(img, mode='RGB')
        img = _trans(img)
        images.append(img.numpy())
    targets = raw_data['y'].reshape(-1)
    targets[np.where(targets == 10)] = 0

    images, targets = torch.tensor(np.stack(images)), torch.tensor(np.array(targets),dtype=torch.long)
    dataset = DigitsDataset(images, targets, aug, con)
    return dataset


def load_mnist_m(root_dir, train=True, aug='', con=0):
    split_list = {
        'train': [
            "mnist_m_train",
            "mnist_m_train_labels.txt"
        ],
        'test': [
            "mnist_m_test",
            "mnist_m_test_labels.txt"
        ],
    }

    split = 'train' if train else 'test'
    data_dir, filename = split_list[split]
    full_path = os.path.join(root_dir, 'MNIST_M', filename)
    data_dir = os.path.join(root_dir, 'MNIST_M', data_dir)
    with open(full_path) as f:
        lines = f.readlines()

    lines = [l.split('\n')[0] for l in lines]
    files = [l.split(' ')[0] for l in lines]
    labels = np.array([int(l.split(' ')[1]) for l in lines]).reshape(-1)
    images = []
    for img in files:
        img = Image.open(os.path.join(data_dir, img)).convert('RGB')
        img = _trans(img)
        images.append(img.numpy())

    images, labels = torch.tensor(np.stack(images)), torch.tensor(labels)
    dataset = DigitsDataset(images, labels, aug, con)
    return dataset

class BatchImageGenerator:
    def __init__(self, flags, stage, file_path, data_loader, b_unfold_label):

        if stage not in ['train', 'test']:
            assert ValueError('invalid stage!')
        self.flags = flags

        self.configuration(flags, stage, file_path)
        self.load_data(data_loader, b_unfold_label)

    def configuration(self, flags, stage, file_path):
        self.batch_size = flags.batch_size
        self.current_index = 0
        self.file_path = file_path
        self.stage = stage

    def load_data(self, data_loader, b_unfold_label):
        file_path = self.file_path
        train = True if self.stage == 'train' else False
       
        self.images, self.labels = data_loader(file_path, train)
        

        if b_unfold_label:
            self.labels = unfold_label(labels=self.labels, classes=len(np.unique(self.labels)))
        assert len(self.images) == len(self.labels)

        self.file_num_train = len(self.labels)
        print('data num loaded:', self.file_num_train)

        if self.stage == 'train':
            self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)

    def get_images_labels_batch(self):
        images = []
        labels = []
        for index in range(self.batch_size):
            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.shuffle()

            images.append(self.images[self.current_index])
            labels.append(self.labels[self.current_index])

            self.current_index += 1

        images = np.stack(images)
        labels = np.stack(labels)

        return images, labels

    def shuffle(self):
        self.file_num_train = len(self.labels)
        self.current_index = 0
        self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)

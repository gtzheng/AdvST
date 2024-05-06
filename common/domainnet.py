
from __future__ import print_function, absolute_import, division
import os
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from common.tools import log, resize_image
from PIL import Image
import torchvision.transforms as transforms

import torch
from common.utils import unfold_label, shuffle_data
from collections import Counter
from common.autoaugment import SVHNPolicy, CIFAR10Policy
from common.randaugment import RandAugment
from functools import partial
from torch.utils.data import Dataset
from torchvision import transforms
from config import DOMANINET_DATA_FOLDER
DOMAINNET_DATA_DIR = DOMANINET_DATA_FOLDER

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

def preprocess_dataset(x, train, img_mean_mode):
    # Compute image mean if applicable
    if img_mean_mode is not None:
        if train:

            if img_mean_mode == "per_channel":
                x_ = np.copy(x)
                x_ = x_.astype('float32') / 255.0
                img_mean = np.array([np.mean(x_[:, :, :, 0]), np.mean(x_[:, :, :, 1]), np.mean(x_[:, :, :, 2])])

            elif img_mean_mode == "imagenet":
                img_mean = np.array([0.485, 0.456, 0.406])

            else:
                raise Exception("Invalid img_mean_mode..!")
            np.save("img_mean.npy", img_mean)

    return x

def load_DomainNet(subset, img_mean_mode="imagenet", data_dir="../../datasets/DomainNet"):
    data_path = os.path.join(data_dir, subset.lower())
    classes = {"airplane": 0, "bicycle": 1, "bus": 2, "car": 3, "horse": 4, "knife": 5, "motorbike": 6,
               "skateboard": 7, "train": 8, "truck": 9}
    img_dim = (224, 224)

    imagedata = []
    labels = []
    for class_dir in classes:
        label = classes[class_dir]
        path = os.path.join(data_path, class_dir)

        for img_file in os.listdir(path):
            if img_file.endswith("jpg") or img_file.endswith("png"):
                img_path = os.path.join(path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = resize_image(img, img_dim,)
                imagedata.append(np.array(img))
                labels.append(label)

    imagedata = np.array(imagedata)
    imagedata = preprocess_dataset(imagedata, train=False, img_mean_mode=img_mean_mode)
    labels = np.array(labels)

    return imagedata, labels

def load_FullDomainNet(subset, train=True, img_mean_mode="imagenet", distillation=False, data_dir="../../datasets"):
    data_path = os.path.join(os.path.dirname(__file__), data_dir)
    img_dim = (224, 224)
    subset = subset.lower()
    if subset == "real":
        labelfile = os.path.join(data_path, "real_train.txt") if train else os.path.join(data_path, "real_test.txt")
    else:
        # labelfile = os.path.join(data_path, "%s.txt" % subset) # os.path.join(data_path, "fast.txt") #
        labelfile = os.path.join(data_path, "{}_train.txt".format(subset)) if train else os.path.join(data_path, "{}_test.txt".format(subset))

    # Gather image paths and labels
    imagepath = []
    labels = []
    with open(labelfile, "r") as f_label:
        for line in f_label:
            temp = line[:-1].split(" ")
            imagepath.append(os.path.join(data_path,temp[0]))
            labels.append(int(temp[1]))
    labels = np.array(labels)
    imagepath = np.array(imagepath)

    return (imagepath, labels)

class DomainNet(Dataset):
    def __init__(self, name, root_folder=DOMAINNET_DATA_DIR, split='train', transform=None, ratio=None):
        if split == 'train':
            train_mode = True
        else:
            train_mode = False
        
        results = load_DomainNet(name, train=train_mode, data_dir=root_folder)
        
        self.x = results[0]
        self.y = results[1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.op_labels = torch.tensor(np.ones(len(self.y),dtype=np.int)*(-1))
        if ratio is not None:
            num = len(self.x)
            indexes = np.random.permutation(num)
            sel_num = int(ratio * num)
            self.x = self.x[indexes[0:sel_num]]
            self.y = self.y[indexes[0:sel_num]]
            self.op_labels = self.op_labels[indexes[0:sel_num]]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        x = transforms.ToPILImage()(x)
        x = self.transform(x)
        op = self.op_labels[index]
        return x, y, op


class DomainNetFull(Dataset):
    def __init__(self, name, root_folder=DOMAINNET_DATA_DIR, split='train', transform=None, ratio=None):
        if split == 'train':
            train_mode = True
        else:
            train_mode = False
        
        results = load_FullDomainNet(name, train=train_mode,  data_dir=root_folder)
        
        self.x = results[0]
        self.y = results[1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.op_labels = torch.tensor(np.ones(len(self.y),dtype=np.int)*(-1))
        if ratio is not None:
            num = len(self.x)
            indexes = np.random.permutation(num)
            sel_num = int(ratio * num)
            self.x = self.x[indexes[0:sel_num]]
            self.y = self.y[indexes[0:sel_num]]
            self.op_labels = self.op_labels[indexes[0:sel_num]]

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

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]

        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)
        return x, y, op

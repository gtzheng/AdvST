from __future__ import print_function, absolute_import, division

import os
import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
from common.pacs import PACS, PACSTensor, PACSMultiple, PACSTensorMultiple, Denormalise
from models.alexnet import alexnet
from models.resnet_vanilla import resnet18
from common.data_reader import BatchImageGenerator
from common.utils import *
from common.utils import (
    fix_all_seed,
    write_log,
    adam,
    sgd,
    compute_accuracy,
    entropy_loss,
    Averager,
)
import pickle
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import kornia
import random
from copy import deepcopy
from common.contrastive import SupConLoss
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import math
from config import PACS_DATA_FOLDER


# https://github.com/HAHA-DL/Episodic-DG
def bn_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


# https://github.com/mil-tokyo/dg_mmld/blob/aef26b2745beabc6356accd183ff3e17f71657ce/util/scheduler.py
from torch.optim.lr_scheduler import _LRScheduler


class inv_lr_scheduler(_LRScheduler):
    def __init__(self, optimizer, alpha, beta, total_epoch, last_epoch=-1):
        self.alpha = alpha
        self.beta = beta
        self.total_epoch = total_epoch
        super(inv_lr_scheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr
            * ((1 + self.alpha * self.last_epoch / self.total_epoch) ** (-self.beta))
            for base_lr in self.base_lrs
        ]


class DataPool:
    def __init__(self, pool_size):
        self.data = [[]] * pool_size
        self.pool_size = pool_size
        self.count = 0
        self.num = 0

    def add(self, x):
        if self.count < self.pool_size:
            self.data[self.count] = x
            self.count += 1
        else:
            self.count = 0
            self.data[self.count] = x
            self.count += 1
        if self.num < self.pool_size:
            self.num += 1

    def get(self, num=-1):
        if self.num == 0:
            return []
        if num < 0:
            return self.data[0 : self.num]
        else:
            num = min(num, self.num)
            indexes = list(range(self.num))
            random.shuffle(indexes)
            sel_indexes = indexes[0:num]
            return [self.data[i] for i in sel_indexes]


def hsv_aug(x, hsv):
    rgb2hsv = kornia.color.RgbToHsv()
    hsv2rgb = kornia.color.HsvToRgb()
    B = x.shape[0]
    hsv_img = rgb2hsv(x) + hsv.view(B, 3, 1, 1)
    rgb_img = hsv2rgb(hsv_img)
    return torch.clamp(rgb_img, -10, 10)


def rotate_aug(x, angle):
    rgb_img = kornia.geometry.transform.rotate(x, torch.clamp(angle, 0.01, 1) * 360)
    return rgb_img


def translate_aug(x, trans):
    h = x.shape[-1] * 0.1
    rgb_img = kornia.geometry.transform.translate(x, torch.clamp(trans, -1, 1) * h)
    return rgb_img


def invert_aug(x, max_val):
    x = torch.clamp(max_val, 0.5, 1.0).view(len(max_val), 1, 1, 1) - x
    return x


def shear_aug(x, val):
    x = kornia.geometry.transform.shear(x, val)
    return x


def contrast_aug(x, con):
    rgb_img = kornia.enhance.adjust_contrast(x, torch.clamp(con, 0.1, 1.9))
    return rgb_img


def sharpness_aug(x, factor):
    x = kornia.enhance.sharpness(x, torch.clamp(factor, 0, 1))
    return x


def scale_aug(x, factor):
    factor = factor.view(len(factor), 1)  # for kornia 0.4
    x = kornia.geometry.transform.scale(x, torch.clamp(factor, 0.5, 2.0))
    return x


def solarize_aug(x, factor):
    x = kornia.enhance.solarize(x, additions=torch.clamp(factor, -0.499, 0.499))
    return x


def equalize_aug(x, factor):
    ex = kornia.enhance.equalize(torch.clamp(x, 0.001, 1.0))
    return ex.detach() + x - x.detach()


def posterize_aug(x, factor):
    bits = torch.randint(0, 9, size=(len(x),)).to(x.device)
    nx = kornia.enhance.posterize(x, bits)
    return nx.detach() + x - x.detach()


def cutout(img_batch, num_holes, hole_size, fill_value=0):
    img_batch = img_batch.clone()
    B = len(img_batch)
    height, width = img_batch.shape[-2:]
    masks = torch.zeros_like(img_batch)
    for _n in range(num_holes):
        if height == hole_size:
            y1 = torch.tensor([0])
        else:
            y1 = torch.randint(0, height - hole_size, (1,))
        if width == hole_size:
            x1 = torch.tensor([0])
        else:
            x1 = torch.randint(0, width - hole_size, (1,))
        y2 = y1 + hole_size
        x2 = x1 + hole_size
        masks[:, :, y1:y2, x1:x2] = 1.0
        # img_batch[:, :, y1:y2, x1:x2] = fill_value
    img_batch = (1.0 - masks) * img_batch + masks * fill_value.view(B, 3, 1, 1)
    return img_batch


def cutout_fixed_num_holes(x, factor, num_holes=8, image_shape=(84, 84)):
    height, width = image_shape
    min_size = min(height, width)
    hole_size = max(int(min_size * 0.2), 1)
    return cutout(x, num_holes=num_holes, hole_size=hole_size, fill_value=factor)


class SemanticAugment(nn.Module):
    def __init__(self, batch_size, op_tuples, op_label):
        super(SemanticAugment, self).__init__()
        self.ops = [op[0] for op in op_tuples]
        self.op_label = op_label
        params = []
        for tup in op_tuples:
            min_val = tup[1][0]
            max_val = tup[1][1]
            num = tup[2]
            init_val = torch.rand(batch_size, num) * (max_val - min_val) + min_val
            init_val = init_val.squeeze(1)
            params.append(torch.nn.Parameter(init_val))
        self.params = nn.ParameterList(params)

    def forward(self, x):
        for i, op in enumerate(self.ops):
            x = torch.clamp(op(x, self.params[i]), 0, 1)
        return x


class Counter:
    def __init__(self):
        self.v = 0
        self.c = 0

    def add(self, x):
        self.v += x
        self.c += 1

    def avg(self):
        if self.c == 0:
            return 0
        return self.v / self.c


class SemanticPerturbation:
    semantics_list = np.array(
        [
            (hsv_aug, (-1, 1), 3),
            (rotate_aug, (0.01, 1), 1),
            (translate_aug, (-1, 1), 2),
            (invert_aug, (0.5, 1), 1),
            (shear_aug, (-0.3, 0.3), 2),
            (contrast_aug, (0.1, 1.9), 1),
            (sharpness_aug, (0, 1), 1),
            (solarize_aug, (-0.5, 0.5), 1),
            (scale_aug, (0.5, 2.0), 1),
            (equalize_aug, (0, 0), 1),
            (posterize_aug, (0, 0), 1),
            (cutout_fixed_num_holes, (0, 1), 3),
        ],
        dtype=object,
    )

    def __init__(self, sel_index=None, max_len=3):
        if sel_index is None:
            self.semantic_aug_list = self.semantics_list
        else:
            self.semantic_aug_list = self.semantics_list[sel_index]
        num = len(self.semantic_aug_list)
        aug_comb = [[c] for c in range(num)]
        num_arr = [num]
        if self.semantic_aug_list[0][0] == hsv_aug:
            op_set = set(list(range(1, num)))
        else:
            op_set = set(list(range(0, num)))
        prev = aug_comb
        for _ in range(2, max_len + 1):
            curr = []
            for comb in prev:
                curr.extend([comb + [c] for c in list(op_set - set(comb))])
            prev = curr
            aug_comb.extend(curr)
            num_arr.append(len(curr))
        probs = []
        for n in num_arr:
            probs.extend([1 / max_len / n] * n)
        self.probs = np.array(probs)
        self.ops = aug_comb

    def sample(self, batch_size):
        op_label = np.random.choice(len(self.ops), p=self.probs / self.probs.sum())
        ops = self.ops[op_label]
        op_tuples = [self.semantic_aug_list[o] for o in ops]
        return SemanticAugment(batch_size, op_tuples, op_label)


class ModelBaseline(object):
    def __init__(self, flags):
        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def get_images(
        self, images, labels, save_path, shuffle=False, sel_indexes=None, nsamples=10
    ):
        class_dict = {}
        for i, l in enumerate(labels):
            if class_dict.get(l, None) is None:
                class_dict[l] = [images[i]]
            else:
                class_dict[l].append(images[i])
        num_classes = len(class_dict)
        total_num_per_class = np.array([len(class_dict[i]) for i in class_dict]).min()
        nsamples = min(nsamples, total_num_per_class)
        indexes = list(range(total_num_per_class))
        if shuffle:
            random.shuffle(indexes)
        if sel_indexes is None:
            sel_indexes = np.array(indexes[0:nsamples])
        else:
            assert nsamples >= len(sel_indexes), "sel_indexes too long"
        data_matrix = []
        keys = sorted(list(class_dict.keys()))
        for c in keys:
            data_matrix.append(np.array(class_dict[c])[sel_indexes])
        data_matrix = np.concatenate(data_matrix, axis=0)
        self.vis_image(data_matrix, nsamples, save_path)
        return sel_indexes

    def vis_image(self, data, max_per_row=10, save_path="./"):
        num = len(data)
        nrow = int(np.ceil(num / max_per_row))

        fig, ax = plt.subplots(figsize=(max_per_row, nrow))
        demo = []
        for i in range(len(data)):
            demo.append(torch.tensor(data[i]))
        demo = torch.stack(demo)

        grid_img = torchvision.utils.make_grid(demo, nrow=max_per_row)
        grid_img = grid_img.permute(1, 2, 0).detach().cpu().numpy()
        ax.imshow(grid_img, interpolation="nearest")
        ax.axis("off")
        fig.savefig(save_path)
        plt.close(fig)

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print("torch.backends.cudnn.deterministic:", torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)
        if flags.model == "resnet18":
            self.network = resnet18(
                pretrained=True,
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            )
        self.network = self.network.cuda()

        print(self.network)
        print("flags:", flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)
        flag_str = (
            "--------Parameters--------\n"
            + "\n".join(["{}={}".format(k, flags.__dict__[k]) for k in flags.__dict__])
            + "\n--------------------"
        )
        print("flags:", flag_str)
        flags_log = os.path.join(flags.logs, "flags_log.txt")
        write_log(flag_str, flags_log)

    def setup_path(self, flags):
        root_folder = PACS_DATA_FOLDER
        dataset_names = ["art_painting", "cartoon", "photo", "sketch"]
        seen_index = flags.seen_index
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        if type(seen_index) == list:
            names = [dataset_names[i] for i in seen_index]
            self.train_name = "+".join(names)
            self.train_dataset = PACSMultiple(root_folder, names, "train")
            self.val_dataset = PACSMultiple(root_folder, names, "val")
            self.test_loaders = []
            for index, name in enumerate(dataset_names):
                if index not in seen_index:
                    dataset = PACS(root_folder, name, "test")
                    loader = DataLoader(
                        dataset,
                        batch_size=flags.batch_size,
                        shuffle=False,
                        num_workers=flags.num_workers,
                        pin_memory=False,
                    )
                    self.test_loaders.append((name, loader))

        else:
            self.train_dataset = PACS(
                root_folder, dataset_names[seen_index], "train", ratio=flags.ratio
            )
            if flags.algorithm == "ERM":
                self.train_dataset.transform = self.train_transform
            self.val_dataset = PACS(root_folder, dataset_names[seen_index], "val")
            self.test_loaders = []
            self.train_name = dataset_names[seen_index]
            for index, name in enumerate(dataset_names):
                if index != seen_index:
                    dataset = PACS(root_folder, name, "test")
                    loader = DataLoader(
                        dataset,
                        batch_size=flags.batch_size,
                        shuffle=False,
                        num_workers=flags.num_workers,
                        pin_memory=False,
                    )

                    self.test_loaders.append((name, loader))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=flags.batch_size,
            shuffle=True,
            num_workers=flags.num_workers,
            pin_memory=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
            pin_memory=False,
        )

    def configure(self, flags):
        for name, param in self.network.named_parameters():
            print(name, param.size())

        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=flags.lr,
            weight_decay=flags.weight_decay,
            momentum=flags.momentum,
            nesterov=True,
        )

        if flags.model == "resnet18":
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, flags.train_epochs * len(self.train_loader)
            )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_per_ele = torch.nn.CrossEntropyLoss(reduction="none")

    def save_model(self, file_name, flags):
        if not os.path.exists(flags.model_path):
            os.makedirs(flags.model_path)
        outfile = os.path.join(flags.model_path, file_name)
        torch.save({"state": self.network.state_dict(), "args": flags}, outfile)

    def train(self, flags):
        os.makedirs(flags.model_path, exist_ok=True)
        best_val_acc = -1
        for epoch in range(flags.train_epochs):
            loss_avger = Averager()
            self.network.train()
            bn_eval(self.network)
            for images_train, labels_train, _ in self.train_loader:
                inputs, labels = images_train.cuda(), labels_train.cuda()
                outputs, _ = self.network(x=inputs)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_avger.add(loss.item(), len(labels_train))
                if flags.model == "resnet18":
                    self.scheduler.step()

            flags_log = os.path.join(flags.logs, "loss_log.txt")

            val_acc = self.batch_test(self.val_loader)
            msg = "[epoch {}] loss {:.4f}, lr {:.4f}, val_acc {:.4f}".format(
                epoch + 1, loss_avger.item(), self.scheduler.get_last_lr()[0], val_acc
            )
            acc_arr = self.batch_test_workflow()
            mean_acc = np.array(acc_arr).mean()
            names = [n for n, _ in self.test_loaders]
            res = (
                "\n{} ".format(self.train_name)
                + " ".join(["{}:{:.6f}".format(n, a) for n, a in zip(names, acc_arr)])
                + " Mean:{:.6f}".format(mean_acc)
            )
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                msg += " (best)"
                self.save_model("best_model_{}.tar".format(self.train_name), flags)
            msg += res
            print(msg)
            write_log(msg, flags_log)
        self.save_model("latest_model_{}.tar".format(self.train_name), flags)

    def test_workflow(self, flags, tag=None):
        accuracies = []
        if tag is not None:
            log_dir = os.path.join(flags.logs, tag)
        else:
            log_dir = flags.logs
        for name, loader in enumerate(self.test_loaders):
            accuracy_test = self.test(
                loader, log_dir=flags.logs, log_prefix="test_{}".format(name)
            )
            accuracies.append(accuracy_test)

        mean_acc = np.mean(accuracies)
        f = open(os.path.join(flags.logs, "acc_test.txt"), mode="a")
        f.write("test accuracy:{}\n".format(mean_acc))
        f.close()

    def test(self, test_loader, log_prefix, log_dir="logs/"):
        # switch on the network test mode
        self.network.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for images_test, labels_test, _ in tqdm(
                test_loader, leave=False, desc="test"
            ):
                images_test, labels_test = images_test.cuda(), labels_test.cuda()
                out, _ = self.network(images_test)
                predictions.append(torch.argmax(out, -1).detach().cpu())
                labels.append(labels_test.detach().cpu())
            predictions = torch.cat(predictions)
            labels = torch.cat(labels)
        accuracy = compute_accuracy(predictions=predictions, labels=labels)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, "{}.txt".format(log_prefix)), mode="a")
        f.write("test accuracy:{}\n".format(accuracy))
        f.close()
        return accuracy

    def batch_test(self, ood_loader):
        # switch on the network test mode
        self.network.eval()
        test_image_preds = []
        test_labels = []
        with torch.no_grad():
            for images_test, labels_test, _ in ood_loader:
                images_test, labels_test = images_test.cuda(), labels_test.cuda()

                out, end_points = self.network(images_test)

                predictions = end_points["Predictions"]
                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)
                test_labels.append(labels_test.cpu().data.numpy())
        predictions = np.concatenate(test_image_preds)
        test_labels = np.concatenate(test_labels)

        accuracy = compute_accuracy(predictions=predictions, labels=test_labels)

        return accuracy

    def batch_test_workflow(self):
        accuracies = []
        with torch.no_grad():
            for name, test_loader in self.test_loaders:
                accuracy_test = self.batch_test(test_loader)
                accuracies.append(accuracy_test)
        return accuracies


class ModelADA(ModelBaseline):
    def __init__(self, flags):
        super(ModelADA, self).__init__(flags)
        # self.num_samples = flags.num_samples

    def configure(self, flags):
        super(ModelADA, self).configure(flags)
        self.dist_fn = torch.nn.MSELoss()

    def setup_path(self, flags):
        super(ModelADA, self).setup_path(flags)
        self.image_denormalise = Denormalise(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self.image_transform = transforms.ToPILImage()

    def maximize(self, flags):
        self.network.eval()
        images, labels = [], []
        self.train_dataset.transform = self.preprocess
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
        )

        for i, (images_train, labels_train, _) in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            leave=False,
            desc="Maximum",
        ):
            inputs, targets = images_train.cuda(), labels_train.cuda()
            out, tuples = self.network(x=inputs)
            inputs_embedding = tuples["Embedding"].detach().clone()
            inputs_embedding.requires_grad_(False)

            inputs_max = inputs.detach().clone()
            inputs_max.requires_grad_(True)
            optimizer = sgd(parameters=[inputs_max], lr=flags.lr_max)

            for ite_max in range(flags.loops_adv):
                out, tuples = self.network(x=inputs_max)
                loss = self.loss_fn(out, targets)
                # loss
                semantic_dist = self.dist_fn(tuples["Embedding"], inputs_embedding)
                loss = (
                    loss - flags.gamma * semantic_dist + flags.eta * entropy_loss(out)
                )

                # init the grad to zeros first
                self.network.zero_grad()
                optimizer.zero_grad()
                (-loss).backward()
                optimizer.step()

            inputs_max = inputs_max.detach().clone().cpu()
            for j in range(len(inputs_max)):
                input_max = self.image_denormalise(inputs_max[j])
                input_max = self.image_transform(input_max.clamp(min=0.0, max=1.0))
                images.append(input_max)
                labels.append(labels_train[j].item())
        images = np.stack(images, 0)
        labels = np.array(labels)
        images = torch.tensor(images).permute(0, 3, 1, 2)
        labels = torch.tensor(labels)
        return images, labels

    def train(self, flags):
        counter_k = 0
        counter_ite = 0
        best_val_acc = -1
        best_avg_acc = -1
        flags_log = os.path.join(flags.logs, "loss_log.txt")
        for epoch in range(0, flags.train_epochs):
            loss_avger = Averager()
            cls_loss_avger = Averager()

            self.network.train()
            bn_eval(self.network)
            self.train_dataset.transform = self.train_transform
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=flags.batch_size,
                shuffle=True,
                num_workers=flags.num_workers,
                pin_memory=True,
            )
            self.scheduler.T_max = counter_ite + len(self.train_loader) * (
                flags.train_epochs - epoch
            )
            for i, (images_train, labels_train, _) in tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                leave=False,
                desc="train-epoch:{}".format(epoch + 1),
            ):
                counter_ite += 1

                inputs, labels = images_train.cuda(), labels_train.cuda()
                # forward with the adapted parameters
                outputs, _ = self.network(x=inputs)
                cls_loss = self.loss_fn(outputs, labels)
                loss = cls_loss - flags.eta_min * entropy_loss(outputs)

                # init the grad to zeros first
                self.optimizer.zero_grad()
                # backward your network
                loss.backward()

                # optimize the parameters
                self.optimizer.step()
                self.scheduler.step()
                loss_avger.add(loss.item())
                cls_loss_avger.add(cls_loss.item())

            val_acc = self.batch_test(self.val_loader)
            acc_arr = self.batch_test_workflow()
            mean_acc = np.array(acc_arr).mean()
            names = [n for n, _ in self.test_loaders]
            res = (
                "\n{} ".format(self.train_name)
                + " ".join(["{}:{:.6f}".format(n, a) for n, a in zip(names, acc_arr)])
                + " Mean:{:.6f}".format(mean_acc)
            )
            msg = "[{}] train_loss:{:.4f} cls:{:.4f} lr:{:.4f} val_acc:{:.4f}".format(
                epoch,
                loss_avger.item(),
                cls_loss_avger.item(),
                self.scheduler.get_last_lr()[0],
                val_acc,
            )

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                msg += " (best)"
                self.save_model("best_model.tar", flags)
            if best_avg_acc < mean_acc:
                best_avg_acc = mean_acc
                self.save_model("best_test_model.tar", flags)
            msg += res
            print(msg)
            write_log(msg, flags_log)

            if ((epoch) % flags.gen_freq == 0) and (
                counter_k < flags.k
            ):  # if T_min iterations are passed
                print("Generating adversarial images [iter {}]".format(counter_k))
                images, labels = self.maximize(flags)
                self.train_dataset.x = torch.cat([self.train_dataset.x, images], dim=0)
                self.train_dataset.y = torch.cat([self.train_dataset.y, labels], dim=0)
                self.train_dataset.op_labels = torch.cat(
                    [self.train_dataset.op_labels, torch.ones_like(labels) * (-1)],
                    dim=0,
                )
                counter_k += 1


class ModelADASemantics(ModelBaseline):
    def __init__(self, flags):
        super(ModelADASemantics, self).__init__(flags)

    def setup_path(self, flags):
        root_folder = PACS_DATA_FOLDER
        dataset_names = ["art_painting", "cartoon", "photo", "sketch"]
        seen_index = flags.seen_index
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        if type(seen_index) == list:
            names = [dataset_names[i] for i in seen_index]
            self.train_name = "+".join(names)
            self.train_dataset = PACSMultiple(root_folder, names, "train")
            self.val_dataset = PACSMultiple(root_folder, names, "val")
            self.test_loaders = []
            for index, name in enumerate(dataset_names):
                if index not in seen_index:
                    dataset = PACS(root_folder, name, "test")
                    loader = DataLoader(
                        dataset,
                        batch_size=flags.batch_size,
                        shuffle=False,
                        num_workers=flags.num_workers,
                        pin_memory=False,
                    )
                    self.test_loaders.append((name, loader))

        else:
            self.train_dataset = PACS(
                root_folder, dataset_names[seen_index], "train", ratio=flags.ratio
            )
            self.train_name = dataset_names[seen_index]
            self.val_dataset = PACS(root_folder, dataset_names[seen_index], "val")
            all_dataset = PACS(
                root_folder, dataset_names[seen_index], "train", ratio=flags.ratio
            )
            all_dataset.x = torch.cat([all_dataset.x, self.val_dataset.x], 0)
            all_dataset.y = torch.cat([all_dataset.y, self.val_dataset.y], 0)
            all_dataset.op_labels = torch.cat(
                [all_dataset.op_labels, self.val_dataset.op_labels], 0
            )
            self.all_loader = DataLoader(
                all_dataset, batch_size=flags.batch_size, num_workers=0, shuffle=False
            )
            self.test_loaders = []
            for index, name in enumerate(dataset_names):
                if index != seen_index:
                    dataset = PACS(root_folder, name, "test")
                    loader = DataLoader(
                        dataset,
                        batch_size=flags.batch_size,
                        shuffle=False,
                        num_workers=flags.num_workers,
                        pin_memory=False,
                    )

                    self.test_loaders.append((name, loader))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=flags.batch_size,
            shuffle=True,
            num_workers=flags.num_workers,
            pin_memory=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
            pin_memory=False,
        )

    def configure(self, flags):
        super(ModelADASemantics, self).configure(flags)
        self.dist_fn = torch.nn.MSELoss()
        self.conloss = SupConLoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.image_transform = transforms.ToPILImage()
        self.image_denormalise = Denormalise(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

        if getattr(flags, "loo", None) is None:
            sel_index = None
        else:
            num_ops = len(SemanticPerturbation.semantics_list)
            sel_index = np.array([i for i in range(num_ops) if i != flags.loo])
        self.semantic_config = SemanticPerturbation(sel_index=sel_index)
        self.scheduler = lr_scheduler.MultiStepLR(
            optimizer=self.optimizer, milestones=[30], gamma=0.1
        )

    def save_model(self, file_name, flags):
        outfile = os.path.join(flags.model_path, file_name)

        aug_probs = [
            (
                "-".join(
                    [
                        self.semantic_config.semantic_aug_list[o][0].__name__
                        for o in self.semantic_config.ops[i]
                    ]
                ),
                self.semantic_config.probs[i],
            )
            for i in range(len(self.semantic_config.ops))
        ]
        torch.save(
            {"state": self.network.state_dict(), "args": flags, "aug_probs": aug_probs},
            outfile,
        )

    def load_model(self, flags):
        print("Load model from ", flags.chkpt_path)
        model_dict = torch.load(flags.chkpt_path)
        prob_tuples = model_dict["aug_probs"]
        self.semantic_config.probs = np.array([t[1] for t in prob_tuples])
        self.network.load_state_dict(model_dict["state"])

    def maximize(self, flags, epoch):
        self.network.eval()
        images, labels, op_labels = [], [], []
        self.train_dataset.transform = self.preprocess
        train_loader = DataLoader(
            self.train_dataset, batch_size=flags.batch_size, num_workers=4, shuffle=True
        )
        mean = self.mean.cuda()
        std = self.std.cuda()
        with tqdm(train_loader, leave=False, total=len(train_loader)) as pbar:
            for ite, (images_train, labels_train, _) in enumerate(pbar):
                inputs, targets = images_train.cuda(), labels_train.cuda()
                out, tuples = self.network(x=inputs)
                inputs_embedding = tuples["Embedding"].data.clone()
                inputs_embedding.requires_grad_(False)

                batch_size = len(inputs)

                semantic_perturb = self.semantic_config.sample(batch_size).to(
                    inputs.device
                )
                op_labels.append(np.array([semantic_perturb.op_label] * batch_size))

                diff_loss = 1.0
                prev_loss = 0.0
                iter_count = 0

                optimizer = torch.optim.RMSprop(
                    semantic_perturb.parameters(), flags.lr_max
                )
                ori_inputs = (
                    inputs * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
                ).data
                while diff_loss > 0.1 and iter_count < flags.loops_adv:
                    inputs_max = semantic_perturb(ori_inputs.data)
                    inputs_max = (inputs_max - mean.view(1, 3, 1, 1)) / std.view(
                        1, 3, 1, 1
                    )

                    out, tuples = self.network(x=inputs_max)
                    cls_loss = self.loss_fn(out, targets)
                    semantic_loss = self.dist_fn(tuples["Embedding"], inputs_embedding)

                    loss = (
                        cls_loss
                        - flags.gamma * semantic_loss
                        + flags.eta * entropy_loss(out)
                    )

                    optimizer.zero_grad()
                    (-loss).backward()

                    optimizer.step()

                    diff_loss = abs((loss - prev_loss).item())
                    prev_loss = loss.item()
                    iter_count += 1

                    pbar.set_postfix(
                        {
                            "loss": "{:.4f}".format(loss.item()),
                            "dist": "{:.6f}".format(semantic_loss.item()),
                        }
                    )

                inputs_max = semantic_perturb(ori_inputs.data)
                inputs_max = (inputs_max - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

                images.append(inputs_max.detach().clone().cpu())
                labels.append(targets.cpu())

        images = torch.cat(images)
        labels = torch.cat(labels)
        op_labels = torch.tensor(np.concatenate(op_labels))
        return images, labels, op_labels

    def train(self, flags):
        counter_k = 0
        best_val_acc = 0
        best_test_acc = 0
        flags_log = os.path.join(flags.logs, "loss_log.txt")
        if not os.path.exists(flags.model_path):
            os.makedirs(flags.model_path)
        train_dataset = deepcopy(self.train_dataset)
        data_pool = DataPool(flags.k + 1)
        data_pool.add((train_dataset.x, train_dataset.y, train_dataset.op_labels))

        # train_dataset.transform = self.train_transform
        train_loader = DataLoader(
            train_dataset,
            batch_size=flags.batch_size,
            num_workers=flags.num_workers,
            shuffle=True
            # sampler=RandomSampler(train_dataset, True,flags.loops_min*flags.batch_size)
        )
        counter_ite = 0
        for epoch in range(1, flags.train_epochs + 1):
            loss_avger = Averager()
            cls_loss_avger = Averager()
            con_loss_avger = Averager()

            for ite, (images_train, labels_train, op_labels) in tqdm(
                enumerate(train_loader, start=1),
                total=len(train_loader),
                leave=False,
                desc="train-epoch{}".format(epoch),
            ):
                self.network.train()
                bn_eval(self.network)
                counter_ite += 1
                inputs, labels = images_train.cuda(), labels_train.cuda()
                img_shape = inputs.shape[-3:]
                outputs, tuples = self.network(x=inputs.reshape(-1, *img_shape))

                cls_loss_ele = self.loss_per_ele(outputs, labels.reshape(-1))
                cls_loss = cls_loss_ele.mean()

                cls_loss_avger.add(cls_loss.item())
                if flags.train_mode == "contrastive":
                    projs = tuples["Projection"]
                    projs = projs.reshape(inputs.shape[0], -1, projs.shape[-1])

                    con_loss = self.conloss(projs, labels)
                    loss = (
                        cls_loss
                        + flags.beta * con_loss
                        - flags.eta_min * entropy_loss(outputs)
                    )
                    con_loss_avger.add(con_loss.item())
                else:
                    loss = cls_loss - flags.eta_min * entropy_loss(outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_avger.add(loss.item())

            self.scheduler.step()

            val_acc = self.batch_test(self.all_loader)
            acc_arr = self.batch_test_workflow()
            mean_acc = np.array(acc_arr).mean()
            names = [n for n, _ in self.test_loaders]
            res = (
                "\n{} ".format(self.train_name)
                + " ".join(["{}:{:.6f}".format(n, a) for n, a in zip(names, acc_arr)])
                + " Mean:{:.6f}".format(mean_acc)
            )
            msg = "[{}] train_loss:{:.4f} cls:{:.4f} con:{:.4f} lr:{:.4f} val_acc:{:.4f}".format(
                epoch,
                loss_avger.item(),
                cls_loss_avger.item(),
                con_loss_avger.item(),
                self.scheduler.get_last_lr()[0],
                val_acc,
            )

            if best_test_acc < mean_acc:
                best_test_acc = mean_acc
                self.save_model("best_test_model.tar", flags)
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                msg += " (best)"
                self.save_model("best_model.tar", flags)
            msg += res
            print(msg)
            write_log(msg, flags_log)

            if (
                epoch % flags.gen_freq == 0
                and epoch < flags.train_epochs
                and counter_k < flags.domain_number
            ):  # if T_min iterations are passed
                print("Generating adversarial images [iter {}]".format(counter_k))
                images, labels, op_labels = self.maximize(flags, epoch)
                data_pool.add((images, labels, op_labels))

                counter_k += 1
            data_batch = data_pool.get()

            gen_x = torch.cat([p[0] for p in data_batch], 0)
            gen_y = torch.cat([p[1] for p in data_batch], 0)
            gen_op_labels = torch.cat([p[2] for p in data_batch], 0)

            train_dataset.x = gen_x
            train_dataset.y = gen_y
            train_dataset.op_labels = gen_op_labels

            train_loader = DataLoader(
                train_dataset,
                batch_size=flags.batch_size,
                num_workers=flags.num_workers,
                shuffle=True
                # sampler=RandomSampler(train_dataset, True,flags.loops_min*flags.batch_size)
            )

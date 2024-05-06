from __future__ import print_function, absolute_import, division

import os
import torch
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
from common.domainnet import Denormalise
from common.domainnet import DomainNetFull
from common.pacs import PACS
from models.resnet_vanilla import resnet18
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
from config import DOMANINET_DATA_FOLDER


# https://github.com/HAHA-DL/Episodic-DG
def bn_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


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
        if self.semantic_aug_list[0][0] == hsv_aug:  # for numerical stability
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

    def get_max_ops(self, topk=5):
        indexes = self.probs.argsort()
        indexes = indexes[::-1]
        max_ops = [self.ops[indexes[i]] for i in range(topk)]
        max_ops = [
            "->".join([self.semantic_aug_list[i][0].__name__ for i in op])
            for op in max_ops
        ]
        return max_ops

    def summarize_ops(self):
        num = len(self.semantic_aug_list)
        op_probs = {i: 0 for i in range(num)}
        for i, ops in enumerate(self.ops):
            for o in ops:
                op_probs[o] += self.probs[i] / len(ops)
        prob_names = {
            t[0].__name__: op_probs[i] for i, t in enumerate(self.semantic_aug_list)
        }
        res_list = [(k, v) for k, v in prob_names.items()]
        res_list = sorted(res_list, key=lambda x: -x[1])
        return res_list


class ModelBaseline(object):
    def __init__(self, flags):
        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def get_images(
        self,
        images,
        labels,
        save_path,
        shuffle=False,
        sel_indexes=None,
        nsamples=10,
        row_wise=True,
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
        if row_wise:
            data_matrix = np.concatenate(data_matrix, axis=0)
            self.vis_image(data_matrix, nsamples, save_path)
        else:
            mat_shape = data_matrix[0].shape[1:]
            data_matrix = np.stack(data_matrix, axis=1).reshape(
                num_classes * nsamples, *mat_shape
            )
            self.vis_image(data_matrix, num_classes, save_path)
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
            print("Use pretrained ResNet-18")
            self.network = resnet18(
                pretrained=True,
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            )
        else:
            raise ValueError("model type not supported")
        self.network = self.network.cuda()

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
        seen_index = flags.seen_index
        dataset_names = [
            "Real",
            "Infograph",
            "Clipart",
            "Painting",
            "Quickdraw",
            "Sketch",
        ]
        root_folder = DOMANINET_DATA_FOLDER
        self.train_name = dataset_names[seen_index]
        self.train_dataset = DomainNetFull(
            dataset_names[seen_index], root_folder=root_folder, split="train"
        )
        self.val_dataset = DomainNetFull(
            dataset_names[seen_index], root_folder=root_folder, split="test"
        )

        if flags.loops_min == -1:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=flags.batch_size,
                shuffle=True,
                num_workers=flags.num_workers,
                pin_memory=False,
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=flags.batch_size,
                shuffle=False,
                num_workers=flags.num_workers,
                pin_memory=False,
                sampler=RandomSampler(
                    self.train_dataset, True, flags.loops_min * flags.batch_size
                ),
            )
        self.all_val_loader = DataLoader(
            self.val_dataset,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
            pin_memory=False,
        )  # 2000 samples for fast validation
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
            pin_memory=False,
            sampler=RandomSampler(self.val_dataset, True, 2000),
        )  # 2000 samples for fast validation

        self.all_loader = self.val_loader
        self.test_loaders = []
        for index, name in enumerate(dataset_names):
            if index != seen_index:
                ood_dataset = DomainNetFull(name, split="test")
                if flags.loops_min == -1:
                    ood_loader = DataLoader(
                        ood_dataset,
                        batch_size=flags.batch_size,
                        num_workers=flags.num_workers,
                        shuffle=False,
                    )
                else:
                    ood_loader = DataLoader(
                        ood_dataset,
                        batch_size=flags.batch_size,
                        num_workers=flags.num_workers,
                        shuffle=False,
                        sampler=RandomSampler(ood_dataset, True, 2000),
                    )
                self.test_loaders.append((name, ood_loader))

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

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
        flags_log = os.path.join(flags.logs, "train_log.txt")
        best_val_acc = -1
        for epoch in range(flags.train_epochs):
            loss_avger = Averager()
            self.network.train()
            for images_train, labels_train, _ in self.train_loader:
                inputs, labels = images_train.cuda(), labels_train.cuda()
                outputs, _ = self.network(x=inputs)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_avger.add(loss.item(), len(labels_train))

                self.scheduler.step()
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
                self.save_model("best_model.tar", flags)
            msg += res
            print(msg)
            write_log(msg, flags_log)
        self.save_model("latest_model.tar", flags)

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

    def configure(self, flags):
        super(ModelADA, self).configure(flags)
        self.image_denormalise = Denormalise(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self.dist_fn = torch.nn.MSELoss()
        self.aug_count = 0

    def setup_path(self, flags):
        super(ModelADA, self).setup_path(flags)
        self.to_image = transforms.ToPILImage()

    def maximize(self, flags):
        self.network.eval()
        images, labels = [], []
        self.aug_count += 1
        if not os.path.exists(flags.aug_folder):
            os.makedirs(flags.aug_folder)
        self.train_dataset.transform = self.train_dataset.preprocess
        if flags.loops_min == -1:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=flags.batch_size,
                shuffle=False,
                num_workers=flags.num_workers,
            )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=flags.batch_size,
                shuffle=False,
                num_workers=flags.num_workers,
                sampler=RandomSampler(
                    self.train_dataset, True, flags.loops_min * flags.batch_size
                ),
            )

        for i, (images_train, labels_train, _) in tqdm(
            enumerate(self.train_loader),
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
                input_max = self.to_image(input_max.clamp(min=0.0, max=1.0))
                name = os.path.join(
                    flags.aug_folder, "augimg_{}-{}-{}.jpg".format(self.aug_count, i, j)
                )
                input_max.save(name)

                images.append(name)
                labels.append(labels_train[j].item())
        return np.array(images), np.array(labels)

    def train(self, flags):
        counter_k = 0
        counter_ite = 0
        best_val_acc = -1
        best_test_acc = -1
        flags_log = os.path.join(flags.logs, "train_log.txt")
        for epoch in range(0, flags.train_epochs):
            loss_avger = Averager()
            cls_loss_avger = Averager()

            self.network.train()
            self.train_dataset.transform = self.train_dataset.train_transform
            if flags.loops_min == -1:
                self.train_loader = torch.utils.data.DataLoader(
                    self.train_dataset,
                    batch_size=flags.batch_size,
                    shuffle=True,
                    num_workers=flags.num_workers,
                    pin_memory=True,
                )
            else:
                self.train_loader = torch.utils.data.DataLoader(
                    self.train_dataset,
                    batch_size=flags.batch_size,
                    shuffle=False,
                    num_workers=flags.num_workers,
                    pin_memory=True,
                    sampler=RandomSampler(
                        self.train_dataset, True, flags.loops_min * flags.batch_size
                    ),
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
                loss = cls_loss  # - flags.eta_min * entropy_loss(outputs)

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
            if best_test_acc < mean_acc:
                best_test_acc = mean_acc
                self.save_model("best_test_model.tar", flags)
            msg += res
            print(msg)
            write_log(msg, flags_log)

            if ((epoch) % flags.gen_freq == 0) and (
                counter_k < flags.k
            ):  # if T_min iterations are passed
                print("Generating adversarial images [iter {}]".format(counter_k))
                images, labels = self.maximize(flags)
                self.train_dataset.x = np.concatenate(
                    [self.train_dataset.x, images], axis=0
                )
                self.train_dataset.y = np.concatenate([self.train_dataset.y, labels])
                self.train_dataset.op_labels = torch.cat(
                    (self.train_dataset.op_labels, torch.tensor([-1] * len(labels)))
                )
                counter_k += 1


class ModelADASemantics(ModelBaseline):
    def __init__(self, flags):
        super(ModelADASemantics, self).__init__(flags)

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
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, flags.train_epochs * flags.loops_min
        )
        self.semantic_config = SemanticPerturbation()

        if getattr(flags, "loo", None) is None:
            sel_index = None
        else:
            num_ops = len(SemanticPerturbation.semantics_list)
            sel_index = np.array([i for i in range(num_ops) if i != flags.loo])
        self.semantic_config = SemanticPerturbation(sel_index=sel_index)

    def maximize(self, flags, epoch):
        self.network.eval()
        images, labels, op_labels = [], [], []
        self.train_dataset.transform = self.train_dataset.preprocess
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=128,
            num_workers=12,
            shuffle=False,
            sampler=RandomSampler(self.train_dataset, True, flags.loops_min * 128),
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
                prev_loss = 1000
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
                    ent_loss = entropy_loss(out)
                    loss = cls_loss - flags.gamma * semantic_loss + flags.eta * ent_loss

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

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)

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
        ori_data = []
        ori_label = []
        ori_op_label = []
        train_loader = DataLoader(
            train_dataset,
            batch_size=flags.batch_size,
            num_workers=flags.num_workers,
            shuffle=False,
            sampler=RandomSampler(
                train_dataset, True, flags.loops_min * flags.batch_size
            ),
        )
        for _, (images_train, labels_train, op_labels) in tqdm(
            enumerate(train_loader, start=1), total=len(train_loader), leave=False
        ):
            ori_data.append(images_train)
            ori_label.append(labels_train)
            ori_op_label.append(op_labels - 1)
        ori_data = torch.cat(ori_data, dim=0)
        ori_label = torch.cat(ori_label, dim=0)
        ori_op_label = torch.cat(ori_op_label, dim=0)
        data_pool.add((ori_data, ori_label, ori_op_label))
        train_loader = DataLoader(
            train_dataset,
            batch_size=flags.batch_size,
            num_workers=flags.num_workers,
            shuffle=False,
            sampler=RandomSampler(
                train_dataset, True, flags.loops_min * flags.batch_size * flags.k
            ),
        )

        counter_ite = 0
        for epoch in range(1, flags.train_epochs + 1):
            loss_avger = Averager()
            cls_loss_avger = Averager()
            con_loss_avger = Averager()
            entloss_avger = Averager()

            for ite, (images_train, labels_train, op_labels) in tqdm(
                enumerate(train_loader, start=1),
                total=len(train_loader),
                leave=False,
                desc="train-epoch{}".format(epoch),
            ):
                self.network.train()
                counter_ite += 1
                inputs, labels = images_train.cuda(), labels_train.cuda()
                img_shape = inputs.shape[-3:]
                outputs, tuples = self.network(x=inputs.reshape(-1, *img_shape))
                cls_loss_ele = self.loss_per_ele(outputs, labels.reshape(-1))
                cls_loss = cls_loss_ele.mean()

                cls_loss_avger.add(cls_loss.item())
                ent_loss = entropy_loss(outputs)
                entloss_avger.add(ent_loss.item())
                if flags.train_mode == "contrastive":
                    projs = tuples["Projection"]
                    projs = projs.reshape(inputs.shape[0], -1, projs.shape[-1])

                    con_loss = self.conloss(projs, labels)

                    loss = cls_loss + flags.beta * con_loss - flags.eta_min * ent_loss
                    con_loss_avger.add(con_loss.item())
                else:
                    loss = cls_loss - flags.eta_min * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_avger.add(loss.item())

                self.scheduler.step()

            val_acc = self.batch_test(self.val_loader)
            acc_arr = self.batch_test_workflow()
            mean_acc = np.array(acc_arr).mean()
            names = [n for n, _ in self.test_loaders]
            res = (
                "\n{} ".format(self.train_name)
                + " ".join(["{}:{:.6f}".format(n, a) for n, a in zip(names, acc_arr)])
                + " Mean:{:.6f}".format(mean_acc)
            )
            msg = "[{}] train_loss:{:.4f} cls:{:.4f} con:{:.4f} ent {:.4f} lr:{:.4f} val_acc:{:.4f}".format(
                epoch,
                loss_avger.item(),
                cls_loss_avger.item(),
                con_loss_avger.item(),
                entloss_avger.item(),
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
            # self.save_model('latest_model.tar', flags)
            if (
                epoch % flags.gen_freq == 0
                and epoch < flags.train_epochs
                and counter_k < flags.domain_number
            ):  # if T_min iterations are passed
                print("Semantic image generation [iter {}]".format(counter_k))
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
                shuffle=False,
                sampler=RandomSampler(
                    train_dataset, True, flags.k * flags.loops_min * flags.batch_size
                ),
            )


class ModelADASemanticsDisk(ModelBaseline):
    def __init__(self, flags):
        super(ModelADASemanticsDisk, self).__init__(flags)

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

    def configure(self, flags):
        super(ModelADASemanticsDisk, self).configure(flags)
        self.dist_fn = torch.nn.MSELoss()
        self.conloss = SupConLoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.image_transform = transforms.ToPILImage()
        self.image_denormalise = Denormalise(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, flags.train_epochs * flags.loops_min
        )
        self.semantic_config = SemanticPerturbation()

        if getattr(flags, "loo", None) is None:
            sel_index = None
        else:
            num_ops = len(SemanticPerturbation.semantics_list)
            sel_index = np.array([i for i in range(num_ops) if i != flags.loo])
        self.semantic_config = SemanticPerturbation(sel_index=sel_index)
        self.aug_count = 0
        if not os.path.exists(flags.aug_folder):
            os.makedirs(flags.aug_folder)
    def maximize(self, flags, epoch):
        self.aug_count = (self.aug_count + 1) % self.pool_size
        self.network.eval()
        images, labels, op_labels = [], [], []
        self.train_dataset.transform = self.train_dataset.preprocess
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=128,
            num_workers=12,
            shuffle=False,
            sampler=RandomSampler(self.train_dataset, True, flags.loops_min * 128),
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
                prev_loss = 1000
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
                    ent_loss = entropy_loss(out)
                    loss = cls_loss - flags.gamma * semantic_loss + flags.eta * ent_loss

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
                for n in range(len(inputs_max)):
                    name = os.path.join(flags.aug_folder, f"aug_{self.aug_count}_{ite}_{n}.pt")
                    img_data = inputs_max[n].detach().clone().cpu()
                    target = targets[n].cpu()
                    torch.save((img_data, target, torch.tensor(semantic_perturb.op_label)), name)
                    
                    images.append(name)
                    labels.append(target.item())
        labels = torch.tensor(labels)

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
        self.pool_size = data_pool.pool_size
        ori_data = []
        ori_label = []
        ori_op_label = []
        train_loader = DataLoader(
            train_dataset,
            batch_size=flags.batch_size,
            num_workers=flags.num_workers,
            shuffle=False,
            sampler=RandomSampler(
                train_dataset, True, flags.loops_min * flags.batch_size
            ),
        )
        for ite, (images_train, labels_train, op_labels) in tqdm(
            enumerate(train_loader, start=1), total=len(train_loader), leave=False
        ):
            
            for n in range(len(images_train)):
                name = os.path.join(flags.aug_folder, f"aug_{self.aug_count}_{ite}_{n}.pt")
                data = images_train[n]
                target = labels_train[n]
                torch.save((data, target, torch.tensor(-2)), name)
                
                ori_data.append(name)
                ori_op_label.append(-2)
            ori_label.append(labels_train)
       
        ori_label = torch.cat(ori_label, dim=0)
        ori_op_label = torch.tensor(ori_op_label)
        data_pool.add((ori_data, ori_label, ori_op_label))
        train_loader = DataLoader(
            train_dataset,
            batch_size=flags.batch_size,
            num_workers=flags.num_workers,
            shuffle=False,
            sampler=RandomSampler(
                train_dataset, True, flags.loops_min * flags.batch_size * flags.k
            ),
        )

        counter_ite = 0
        for epoch in range(1, flags.train_epochs + 1):
            loss_avger = Averager()
            cls_loss_avger = Averager()
            con_loss_avger = Averager()
            entloss_avger = Averager()

            for ite, (images_train, labels_train, op_labels) in tqdm(
                enumerate(train_loader, start=1),
                total=len(train_loader),
                leave=False,
                desc="train-epoch{}".format(epoch),
            ):
                self.network.train()
                counter_ite += 1
                inputs, labels = images_train.cuda(), labels_train.cuda()
                img_shape = inputs.shape[-3:]
                outputs, tuples = self.network(x=inputs.reshape(-1, *img_shape))
                cls_loss_ele = self.loss_per_ele(outputs, labels.reshape(-1))
                cls_loss = cls_loss_ele.mean()

                cls_loss_avger.add(cls_loss.item())
                ent_loss = entropy_loss(outputs)
                entloss_avger.add(ent_loss.item())
                if flags.train_mode == "contrastive":
                    projs = tuples["Projection"]
                    projs = projs.reshape(inputs.shape[0], -1, projs.shape[-1])

                    con_loss = self.conloss(projs, labels)

                    loss = cls_loss + flags.beta * con_loss - flags.eta_min * ent_loss
                    con_loss_avger.add(con_loss.item())
                else:
                    loss = cls_loss - flags.eta_min * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_avger.add(loss.item())

                self.scheduler.step()

            val_acc = self.batch_test(self.val_loader)
            acc_arr = self.batch_test_workflow()
            mean_acc = np.array(acc_arr).mean()
            names = [n for n, _ in self.test_loaders]
            res = (
                "\n{} ".format(self.train_name)
                + " ".join(["{}:{:.6f}".format(n, a) for n, a in zip(names, acc_arr)])
                + " Mean:{:.6f}".format(mean_acc)
            )
            msg = "[{}] train_loss:{:.4f} cls:{:.4f} con:{:.4f} ent {:.4f} lr:{:.4f} val_acc:{:.4f}".format(
                epoch,
                loss_avger.item(),
                cls_loss_avger.item(),
                con_loss_avger.item(),
                entloss_avger.item(),
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
            # self.save_model('latest_model.tar', flags)
            if (
                epoch % flags.gen_freq == 0
                and epoch < flags.train_epochs
                and counter_k < flags.domain_number
            ):  # if T_min iterations are passed
                print("Semantic image generation [iter {}]".format(counter_k))
                images, labels, op_labels = self.maximize(flags, epoch)
                data_pool.add((images, labels, op_labels))

                counter_k += 1
            data_batch = data_pool.get()

            gen_x = [name for p in data_batch for name in p[0]]
            gen_y = torch.cat([p[1] for p in data_batch], 0)
            gen_op_labels = torch.cat([p[2] for p in data_batch], 0)

            train_dataset.x = gen_x
            train_dataset.y = gen_y
            train_dataset.op_labels = gen_op_labels
            train_loader = DataLoader(
                train_dataset,
                batch_size=flags.batch_size,
                num_workers=flags.num_workers,
                shuffle=False,
                sampler=RandomSampler(
                    train_dataset, True, flags.k * flags.loops_min * flags.batch_size
                ),
            )
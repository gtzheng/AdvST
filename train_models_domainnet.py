from main_domainnet import main as train_domainnet
import argparse
import os
import subprocess
import multiprocessing as mp
from functools import partial, reduce
import shlex
import time
from copy import deepcopy
import glob
import shutil
from common.utils import get_freer_gpu
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def combine_all(list_of_lists):
    if len(list_of_lists) == 1:
        return [[n] for n in list_of_lists[0]]
    first_list = list_of_lists[0]
    sub_list_of_lists = list_of_lists[1:]
    new_list = combine_all(sub_list_of_lists)
    merged_list = []
    for item in first_list:
        for nl in new_list:
            merged_list.append([item] + nl)
    return merged_list


def main(args):
    save_path = args.save_path

    # For AdvST/AdvST-ME experiments
    domainnet_AdvST_args = Namespace(
        seed=args.seed,
        algorithm="AdvST",  # AdvST
        model="resnet18",  # resnet18
        batch_size=128,
        num_classes=345,
        seen_index=0,
        train_epochs=200,
        loops_min=300,  # number of batches per epoch
        loops_adv=30,
        dataset="domainnet",
        lr=1e-3,
        lr_max=1.0,
        momentum=0.9,
        weight_decay=5e-5,
        path=save_path,
        deterministic=True,
        k=2,
        gamma=10,  # corresponds to lambda in the paper
        eta=10.0,  # set to a nonzero value for AdvST-ME, parameter for the regularizer in the maximization procedure
        eta_min=0.1,  # parameter for the entropy regularizer in the minimization procedure
        beta=1.0,  # paramter for the contrastive loss regularizer
        gpu=args.gpu,
        num_workers=8,
        train_mode="contrastive",  # contrastive, norm
        tag="",
        gen_freq=1,
        domain_number=500,
        ratio=1.0,
    )
    # For ADA/ME-ADA/ERM experiments
    domainnet_ada_args = Namespace(
        seed=args.seed,
        algorithm="ADA",  # ERM, ADA
        model="resnet18",  # alexnet or resnet18
        batch_size=128,
        num_classes=345,
        seen_index=0,
        train_epochs=50,
        loops_min=-1,  # use all training data per epoch
        loops_adv=50,
        dataset="domainnet",
        lr=0.001,
        lr_max=50.0,
        momentum=0.9,
        weight_decay=5e-5,
        path=save_path,
        deterministic=True,
        k=3,
        gamma=10,
        eta=10.0,
        eta_min=0.0,
        beta=0.0,
        gpu=args.gpu,
        num_workers=4,
        ratio=1.0,
        train_mode="norm",  # contrastive, norm
        tag="",
        gen_freq=1,
        domain_number=3,
        aug_folder="/root/folder/ME-ADA/aug_data_me"  # store generated images to disk instead of memory
        # aug=''
    )

    expr_args = domainnet_ada_args
    train_domainnet(expr_args)


if __name__ == "__main__":
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1, help="")
    train_arg_parser.add_argument(
        "--save_path",
        type=str,
        default="domainnet_experiments/test",
        help="path to saved models and results",
    )
    args = train_arg_parser.parse_args()
    gpu = ",".join([str(i) for i in get_freer_gpu()[0:1]])
    args.gpu = gpu
    main(args)

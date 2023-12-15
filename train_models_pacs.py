from main_pacs import main as train_pacs
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


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(args):
    save_path = args.save_path
    # For experiments on AdvST/AdvST-ME
    pacs_AdvST_args = Namespace(
        seed=args.seed,
        algorithm="AdvST",  # ERM, ADA, AdvST
        model="resnet18",
        batch_size=32,
        num_classes=7,
        seen_index=2,  # photo domain ['art_painting','cartoon','photo','sketch']
        train_epochs=50,
        loops_min=100,  # number of batches per epoch
        loops_adv=50,
        dataset="pacs",
        lr=0.001,
        lr_max=5.0,
        momentum=0.9,
        weight_decay=5e-5,
        path=save_path,
        deterministic=True,
        k=3,
        gamma=10,  # corresponds to lambda in the paper
        eta=10.0,  # set to a nonzero value for AdvST-ME, parameter for the regularizer in the maximization procedure
        eta_min=0.01,  # parameter for the entropy regularizer in the minimization procedure
        beta=1.0,  # paramter for the contrastive loss regularizer
        gpu=args.gpu,
        num_workers=8,
        train_mode="contrastive",  # contrastive, norm
        tag="",
        gen_freq=1,
        domain_number=100,
        ratio=1.0,  # select how much training data to use
    )
    # For ADA/ME-ADA/ERM experiments
    pacs_ada_args = Namespace(
        seed=args.seed,
        algorithm="ADA",  # ERM, ADA
        model="resnet18",
        batch_size=32,
        num_classes=7,
        seen_index=2,
        train_epochs=50,
        loops_min=-1,  # use all training data per epoch
        loops_adv=50,
        dataset="pacs",
        lr=0.001,
        lr_max=50.0,
        momentum=0.9,
        weight_decay=5e-5,
        path=save_path,
        deterministic=True,
        k=3,
        gamma=10,
        eta=0.0,
        eta_min=0.0,
        beta=0.0,
        gpu=args.gpu,
        num_workers=4,
        train_mode="norm",  # contrastive, norm
        tag="",
        gen_freq=1,
        ratio=1.0,
    )

    expr_args = pacs_AdvST_args
    train_pacs(expr_args)


if __name__ == "__main__":
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1, help="")
    train_arg_parser.add_argument(
        "--save_path",
        type=str,
        default="pacs_experiments/pacs_AdvST_test",
        help="path to saved models and results",
    )
    args = train_arg_parser.parse_args()
    gpu = ",".join([str(i) for i in get_freer_gpu()[0:1]])
    args.gpu = gpu
    main(args)

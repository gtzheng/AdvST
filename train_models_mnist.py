from main_mnist import main as train_mnist
import argparse
import os
import numpy as np
from common.utils import get_freer_gpu
from model_mnist import SemanticPerturbation


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(args):
    save_path = args.save_path
    # For AdvST/AdvST-ME experiments
    mnist_AdvST_args = Namespace(
        seed=args.seed,
        algorithm="AdvST",
        test_every=100,
        batch_size=32,
        num_classes=10,
        num_samples=10000,
        step_size=25,
        loops_train=10000,
        loops_min=100,
        loops_adv=20,
        seen_index=0,
        lr=0.0001,
        lr_max=0.2,
        weight_decay=0.0,
        path=save_path,
        deterministic=True,
        balanced_weight=False,
        imbalanced_class=False,
        imbalance_ratio=2.0,
        store_data=False,
        k=5,  # domain pool size
        gamma=100,  # corresponds to lambda in the paper
        eta=0.0,  # set to a nonzero value for AdvST-ME, parameter for the regularizer in the maximization procedure
        eta_min=10.0,  # parameter for the entropy regularizer in the minimization procedure
        beta=1.0,  # paramter for the contrastive loss regularizer
        gpu=args.gpu,
        num_workers=4,
        train_mode="contrastive",  # contrastive, norm
        tag="",
        aug_number=0,
        gen_freq=1,
        domain_number=200,
        train_epochs=50,
        aug="",
    )

    # For ADA/ME-ADA/ERM experiment
    mnist_ada_args = Namespace(
        seed=args.seed,
        algorithm="ADA",  # ADA, ERM
        test_every=100,
        batch_size=32,
        num_classes=10,
        num_samples=10000,
        step_size=100,
        loops_train=10000,
        loops_min=100,
        loops_adv=20,
        seen_index=0,
        lr=0.0001,
        lr_max=1.0,
        weight_decay=0.0,
        path=save_path,
        deterministic=True,
        balanced_weight=False,
        imbalanced_class=False,
        imbalance_ratio=2.0,
        store_data=False,
        k=3,
        gamma=100,
        eta=0.0,  # set to a nonzero value for ME-ADA
        beta=0.0,
        gpu=args.gpu,
        num_workers=4,
        train_mode="norm",  # contrastive, norm
        tag="",
        aug_number=0,
        gen_freq=1,
        train_epochs=200,
        aug="",
    )

    mnist_args = mnist_AdvST_args
    train_mnist(mnist_args)


if __name__ == "__main__":
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1, help="")
    train_arg_parser.add_argument(
        "--save_path",
        type=str,
        default="mnist_AdvST_test",
        help="path to saved models and results",
    )
    args = train_arg_parser.parse_args()
    gpu = ",".join([str(i) for i in get_freer_gpu()[0:1]])
    args.gpu = gpu
    main(args)

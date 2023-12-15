from __future__ import print_function, absolute_import

import argparse
import os
from model_mnist import ModelBaseline, ModelADA, ModelADASemantics


from common.utils import time_str, Timer, set_gpu
from test_models import main as eval_models


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(args):
    timer = Timer()
    args.model_path = os.path.join(args.path, "models")
    args.logs = os.path.join(args.path, "logs")
    set_gpu(args.gpu)
    if args.algorithm == "ERM":
        model_obj = ModelBaseline(flags=args)
    elif args.algorithm == "ADA":
        model_obj = ModelADA(flags=args)
    elif args.algorithm == "AdvST":
        model_obj = ModelADASemantics(flags=args)
    else:
        raise RuntimeError

    model_obj.train(flags=args)
    expr_info = "{}-gamma:{:.4f}-lr_max:{:.4f}-eta:{:.4f}".format(
        args.algorithm, args.gamma, args.lr_max, args.eta
    )
    if len(args.tag) > 0:
        expr_info += "-{}".format(args.tag)
    eval_args = Namespace(gpu=args.gpu, path=args.path, info=expr_info, dataset="mnist")
    eval_models(eval_args)
    elapsed_time = timer.t()
    print("Elapsed time {}".format(time_str(elapsed_time)))


if __name__ == "__main__":
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1, help="")
    train_arg_parser.add_argument(
        "--algorithm", type=str, default="ModelFADA", help="Choose algorithm."
    )
    train_arg_parser.add_argument("--test_every", type=int, default=50, help="")
    train_arg_parser.add_argument("--batch_size", type=int, default=32, help="")
    train_arg_parser.add_argument("--num_classes", type=int, default=10, help="")
    train_arg_parser.add_argument("--num_samples", type=int, default=10000, help="")
    train_arg_parser.add_argument("--step_size", type=int, default=10001, help="")
    train_arg_parser.add_argument("--loops_train", type=int, default=10000, help="")
    train_arg_parser.add_argument("--loops_min", type=int, default=100, help="")
    train_arg_parser.add_argument("--loops_adv", type=int, default=20, help="")
    train_arg_parser.add_argument("--seen_index", type=int, default=0, help="")
    train_arg_parser.add_argument("--lr", type=float, default=0.0001, help="")
    train_arg_parser.add_argument("--lr_max", type=float, default=1.0, help="")
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.0, help="")
    train_arg_parser.add_argument(
        "--path", type=str, default="mnist_aug_every_batch", help=""
    )
    train_arg_parser.add_argument("--deterministic", type=bool, default=True, help="")
    train_arg_parser.add_argument(
        "--balanced_weight", type=bool, default=False, help=""
    )
    train_arg_parser.add_argument(
        "--imbalanced_class", type=bool, default=False, help=""
    )
    train_arg_parser.add_argument("--imbalance_ratio", type=float, default=2.0, help="")
    train_arg_parser.add_argument("--store_data", type=bool, default=False, help="")
    train_arg_parser.add_argument("--k", type=int, default=5, help="")
    train_arg_parser.add_argument("--domain_number", type=int, default=2, help="")
    train_arg_parser.add_argument("--gen_freq", type=int, default=2, help="")
    train_arg_parser.add_argument("--num_workers", type=int, default=0, help="")
    train_arg_parser.add_argument("--train_epochs", type=int, default=100, help="")
    train_arg_parser.add_argument("--gamma", type=float, default=1.0, help="")
    train_arg_parser.add_argument("--eta", type=float, default=1.0, help="")
    train_arg_parser.add_argument("--eta_min", type=float, default=0.0, help="")
    train_arg_parser.add_argument("--beta", type=float, default=1.0, help="")
    train_arg_parser.add_argument(
        "--train_mode", type=str, default="contrastive", help=""
    )
    train_arg_parser.add_argument("--gpu", type=str, default="0", help="")
    train_arg_parser.add_argument("--tag", type=str, default="", help="")
    train_arg_parser.add_argument("--loo", type=int, default=10, help="")
    train_arg_parser.add_argument("--aug_number", type=int, default=0, help="")
    train_arg_parser.add_argument("--aug", type=str, default="", help="")
    args = train_arg_parser.parse_args()
    main(args)

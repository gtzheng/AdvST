import argparse
import os
from model_pacs import ModelBaseline, ModelADA, ModelADASemantics
from common.utils import time_str, Timer, set_gpu
from test_models import main as eval_models


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(args):
    set_gpu(args.gpu)
    args.model_path = os.path.join(args.path, "models")
    args.logs = os.path.join(args.path, "logs")

    if args.algorithm == "ERM":
        model_obj = ModelBaseline(flags=args)
    elif args.algorithm == "ADA":
        model_obj = ModelADA(flags=args)
    elif args.algorithm == "AdvST":
        model_obj = ModelADASemantics(flags=args)
    else:
        raise RuntimeError
    timer = Timer()
    model_obj.train(flags=args)
    expr_info = "Evaluation-gamma:{:.4f}-lr_max:{:.4f}-eta:{:.4f}".format(
        args.gamma, args.lr_max, args.eta
    )
    if len(args.tag) > 0:
        expr_info += "-{}".format(args.tag)
    eval_args = Namespace(gpu=args.gpu, path=args.path, info=expr_info, dataset="pacs")
    eval_models(eval_args)
    elapsed_time = timer.t()
    print("Elapsed time {}".format(time_str(elapsed_time)))


if __name__ == "__main__":
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1, help="")
    train_arg_parser.add_argument(
        "--algorithm", type=str, default="AdvST", help="Choose algorithm."
    )
    train_arg_parser.add_argument(
        "--model", type=str, default="alexnet", help="Choose model."
    )
    train_arg_parser.add_argument("--test_every", type=int, default=50, help="")
    train_arg_parser.add_argument("--batch_size", type=int, default=32, help="")
    train_arg_parser.add_argument("--num_classes", type=int, default=7, help="")
    train_arg_parser.add_argument("--step_size", type=int, default=3001, help="")
    train_arg_parser.add_argument("--bn_eval", type=int, default=0, help="")
    train_arg_parser.add_argument("--train_epochs", type=int, default=10001, help="")
    train_arg_parser.add_argument(
        "--seen_index", default=[1, 2, 3], nargs="+", type=int, help=""
    )
    train_arg_parser.add_argument("--lr", type=float, default=0.001, help="")
    train_arg_parser.add_argument("--lr_max", type=float, default=1.0, help="")

    train_arg_parser.add_argument(
        "--weight_decay", type=float, default=0.00005, help=""
    )
    train_arg_parser.add_argument("--momentum", type=float, default=0.9, help="")
    train_arg_parser.add_argument("--path", type=str, default="", help="")
    train_arg_parser.add_argument(
        "--state_dict",
        type=str,
        default="https://download.pytorch.org/models/resnet18-5c106cde.pth",
        help="",
    )
    train_arg_parser.add_argument("--loops_adv", type=int, default=50, help="")
    train_arg_parser.add_argument("--loops_min", type=int, default=100, help="")
    train_arg_parser.add_argument("--deterministic", type=bool, default=False, help="")
    train_arg_parser.add_argument(
        "--balanced_weight", type=bool, default=False, help=""
    )
    train_arg_parser.add_argument(
        "--imbalanced_class", type=bool, default=True, help=""
    )
    train_arg_parser.add_argument("--imbalance_ratio", type=float, default=2.0, help="")
    train_arg_parser.add_argument("--store_data", type=bool, default=False, help="")
    train_arg_parser.add_argument("--k", type=int, default=10, help="")
    train_arg_parser.add_argument("--gamma", type=float, default=10.0, help="")
    train_arg_parser.add_argument("--train_mode", type=str, default="normal", help="")
    train_arg_parser.add_argument("--dataset", type=str, default="pacs", help="")
    train_arg_parser.add_argument("--beta", type=float, default=1.0, help="")
    train_arg_parser.add_argument("--eta", type=float, default=0.0, help="")
    train_arg_parser.add_argument("--eta_min", type=float, default=0.0, help="")
    train_arg_parser.add_argument("--gpu", type=str, default="0", help="")
    train_arg_parser.add_argument("--tag", type=str, default="", help="")
    train_arg_parser.add_argument("--domain_number", type=int, default=2, help="")
    train_arg_parser.add_argument("--gen_freq", type=int, default=2, help="")
    train_arg_parser.add_argument("--num_workers", type=int, default=4, help="")
    args = train_arg_parser.parse_args()

    main(args)

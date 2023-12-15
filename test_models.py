import argparse

from model_mnist import ModelBaseline as ERM_MNIST
from model_mnist import ModelADA as ADA_MNIST
from model_mnist import ModelADASemantics as AdvST_MNIST


from model_pacs import ModelBaseline as ERM_PACS
from model_pacs import ModelADA as ADA_PACS
from model_pacs import ModelADASemantics as AdvST_PACS

from model_domainnet import ModelBaseline as ERM_DomainNet
from model_domainnet import ModelADA as ADA_DomainNet
from model_domainnet import ModelADASemantics as AdvST_DomainNet

from common.utils import time_str, Timer, set_gpu
import glob
import os
import torch
import numpy as np


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(args):
    timer = Timer()

    set_gpu(args.gpu)

    def load_model(path2model, dataset):
        model_ckpt = torch.load(path2model)
        args = model_ckpt["args"]
        if dataset == "mnist":
            if args.algorithm == "ERM":
                model_obj = ERM_MNIST(flags=args)
            elif args.algorithm == "ADA":
                model_obj = ADA_MNIST(flags=args)
            elif args.algorithm == "AdvST":
                model_obj = AdvST_MNIST(flags=args)
        elif dataset == "pacs":
            if args.algorithm == "ERM":
                model_obj = ERM_PACS(flags=args)
            elif args.algorithm == "ADA":
                model_obj = ADA_PACS(flags=args)
            elif args.algorithm == "AdvST":
                model_obj = AdvST_PACS(flags=args)

        elif dataset == "domainnet":
            args.loops_min = -1  # use all test data for evaluation
            if args.algorithm == "ERM":
                model_obj = ERM_DomainNet(flags=args)
            elif args.algorithm == "ADA":
                model_obj = ADA_DomainNet(flags=args)
            elif args.algorithm == "AdvST":
                model_obj = AdvST_DomainNet(flags=args)

        else:
            raise RuntimeError
        model_obj.network.load_state_dict(model_ckpt["state"])
        return model_obj

    model_list = []
    last_model_path = os.path.join(args.path, "models", "best_test_model.tar")
    best_model_path = os.path.join(args.path, "models", "best_model.tar")
    model_list.append(last_model_path)
    model_list.append(best_model_path)

    with open(os.path.join(args.path, "results.txt"), "a") as fout:
        fout.write(args.info + "\n")
        for idx, mpath in enumerate(model_list):
            if args.dataset == "mnist":
                model_obj = load_model(mpath, "mnist")
                acc_arr = model_obj.batch_test_workflow()
                mean_acc = np.array(acc_arr).mean()
                if idx == len(model_list) - 2:
                    tag = "last"
                elif idx == len(model_list) - 1:
                    tag = "best"
                else:
                    tag = idx
                res = (
                    "{} ".format(tag)
                    + " ".join(["{:.6f}".format(a) for a in acc_arr])
                    + " {:.6f}".format(mean_acc)
                )
                print(res)
                fout.write(res + "\n")
            elif args.dataset == "pacs":
                model_obj = load_model(mpath, "pacs")
                acc_arr = model_obj.batch_test_workflow()
                mean_acc = np.array(acc_arr).mean()
                if idx == len(model_list) - 2:
                    tag = "last"
                elif idx == len(model_list) - 1:
                    tag = "best"
                else:
                    tag = idx

                names = [n for n, _ in model_obj.test_loaders]
                res = (
                    "{}:{} ".format(tag, model_obj.train_name)
                    + " ".join(
                        ["{}:{:.6f}".format(n, a) for n, a in zip(names, acc_arr)]
                    )
                    + " Mean:{:.6f}".format(mean_acc)
                )

                print(res)
                fout.write(res + "\n")

            elif args.dataset == "domainnet":
                model_obj = load_model(mpath, "domainnet")
                acc_arr = model_obj.batch_test_workflow()
                val_acc = model_obj.batch_test(model_obj.all_val_loader)
                mean_acc = np.array(acc_arr).mean()
                if idx == len(model_list) - 2:
                    tag = "last"
                elif idx == len(model_list) - 1:
                    tag = "best"
                else:
                    tag = idx

                names = [n for n, _ in model_obj.test_loaders]
                res = (
                    "{}:{} ".format(tag, model_obj.train_name)
                    + " ".join(
                        ["{}:{:.6f}".format(n, a) for n, a in zip(names, acc_arr)]
                    )
                    + " Mean:{:.6f}".format(mean_acc)
                )
                res += "\n val acc {:.4f}".format(val_acc)
                print(res)
                fout.write(res + "\n")
    elapsed_time = timer.t()
    print("Elapsed time {}".format(time_str(elapsed_time)))


if __name__ == "__main__":
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--gpu", type=str, default="0", help="")
    train_arg_parser.add_argument(
        "--path", type=str, default="/path/to/save/folder", help=""
    )
    train_arg_parser.add_argument("--info", type=str, default="Evaluation", help="")
    train_arg_parser.add_argument("--dataset", type=str, default="mnist", help="")
    args = train_arg_parser.parse_args()
    main(args)

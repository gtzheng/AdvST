# Introduction
This is the code for the AAAI 2024 Paper _AdvST: Revisiting Data Augmentations for Single Domain Generalization_.

# Data

## Digits Dataset
Download the MNIST-M dataset from https://drive.google.com/drive/folders/0B_tExHiYS-0vR2dNZEU4NGlSSW8, rename the folder as MNIST_M.

Download the SYN dataset from https://drive.google.com/file/d/0B9Z4d7lAwbnTSVR1dEFSRUFxOUU/view, rename the folder as SYN.

Move the MNIST_M and SYN folders to the same folder DIGITS_DATA_FOLDER which is configured in `config.py`.

MNIST, SVHN and USPS will be automatically downloaded to DIGITS_DATA_FOLDER.


## PACS Dataset

Download the PACS dataset (h5py files pre-read) from https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ, rename the folder as PACS.

Use the path of the PACS folder as the value of PACS_DATA_FOLDER in `config.py`.

## DomainNet dataset

Download [DomainNet (clean version)](https://ai.bu.edu/M3SDA/)


Create a new `DomainNet` folder

Extract each domain's zip file under its respective subfolder (For example: `datasets/DomainNet/clipart`)

Use the path of the DomainNet folder as the value of DOMANINET_DATA_FOLDER in config.py.

# Experiments

## Digits Dataset
```
python train_models_mnist.py --save_path ./mnist_experiments/AdvST_experiment
```



## PACS Dataset
```
python train_models_pacs.py --save_path ./pacs_experiments/AdvST_experiment
```

## DomainNet Dataset
```
python train_models_domainnet.py --save_path ./domainnet_experiments/AdvST_experiment
```

# Citation

Please consider citing this paper if you find the code helpful.
```
@inproceedings{zhengAAAI24AdvST,
  title={AdvST: Revisiting Data Augmentations for Single Domain Generalization},
  author={Zheng, Guangtao and Huai, Mengdi and Zhang, Aidong},
  booktitle={The 38th Annual AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
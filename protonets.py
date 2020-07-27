import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

import config
from models import MnistModel
from utils import create_dirs, fix_seeds
from datasets import OmniglotDataset
from samplers import FewShotBatchSampler


def parse_args():
    usage = f'Usage: python {__file__} [-t | --train] [-p | --predict]'
    parser = ArgumentParser(usage=usage)
    parser.add_argument('-t', '--train', action='store_true', help='train mnist model')
    parser.add_argument('-p', '--predict', action='store_true', help='load model and demonstrate prediction')
    args = parser.parse_args()
    return args, parser


def train():
    pass


def predict():
    pass


def test():
    n, k, q = 1, 3, 5
    dataset = OmniglotDataset(subset='background')
    sampler = FewShotBatchSampler(dataset,
                                  episodes_per_epoch=100,
                                  n=n,
                                  k=k,
                                  q=q,
                                  num_tasks=1)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
    for i, batch in enumerate(dataloader):
        # x.shape: [k * (n + q) * num_tasks, 1, 105, 105]
        # y.shape: [k * q]
        x, y = prepare_batch(batch, k, q)
        break


def prepare_batch(batch, k, q):
    # data.shape: [batch_size, channels, height, width] = [k * (n + q) * num_tasks, 1, 105, 105]
    # label.shape: [k * (n + q) * num_tasks] 
    # k * (n + q) means [n support sets of class 1 to k, q query sets of class 1 to k]
    data, label = batch

    x = data.double() # .cuda()

    # y = [*([0] * q), *([1] * q), ..., *([k - 1] * q)]
    # 正解ラベルはq個ごとにまとまっているため，q個ごとにclassification用のラベルを[0, k)で振り直す
    y = torch.arange(0, k, 1 / q).long() # .cuda()

    return x, y


if __name__ == '__main__':
    args, parser = parse_args()

    fix_seeds(0)
    create_dirs(config.BASE_PATH)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test()
    exit()

    if args.train:
        train()
    elif args.predict:
        predict()
    else:
        parser.print_help()
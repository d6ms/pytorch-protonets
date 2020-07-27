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
    dataset = OmniglotDataset(subset='background')
    sampler = FewShotBatchSampler(dataset,
                                  episodes_per_epoch=100,
                                  n=1,
                                  k=2,
                                  q=4,
                                  num_tasks=1)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
    for data, label in dataloader:
        print(data.shape)
        break

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
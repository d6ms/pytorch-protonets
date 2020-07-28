import os
from argparse import ArgumentParser

import torch

import config
from utils import create_dirs, fix_seeds
from train import train


def parse_args():
    usage = f'Usage: python {__file__} [-t | --train] [-p | --predict]'
    parser = ArgumentParser(usage=usage)
    parser.add_argument('-t', '--train', action='store_true', help='train model')
    parser.add_argument('-p', '--predict', action='store_true', help='load model and demonstrate prediction')
    args = parser.parse_args()
    return args, parser


if __name__ == '__main__':
    args, parser = parse_args()

    fix_seeds(0)
    create_dirs(config.BASE_PATH)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # n, k, q = 1, 60, 5
    train(3, 1, 3, 5, episodes_per_epoch=5)
    exit()

    if args.train:
        train()
    else:
        parser.print_help()
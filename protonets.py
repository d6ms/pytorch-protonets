import os
from argparse import ArgumentParser

import torch

import config
from models import MnistModel
from utils import create_dirs
from datasets import OmniglotDataset

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

if __name__ == '__main__':
    args, parser = parse_args()

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
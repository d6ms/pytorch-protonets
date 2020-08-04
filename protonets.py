import os
from argparse import ArgumentParser

import torch

import config
from utils import create_dirs, fix_seeds
from train import train


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train model')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--episodes', default=100, type=int)
    parser.add_argument('--n-train', default=1, type=int)
    parser.add_argument('--n-eval', default=1, type=int)
    parser.add_argument('--k-train', default=60, type=int)
    parser.add_argument('--k-eval', default=5, type=int)
    parser.add_argument('--q-train', default=5, type=int)
    parser.add_argument('--q-eval', default=1, type=int)
    args = parser.parse_args()
    return args, parser


if __name__ == '__main__':
    args, parser = parse_args()

    fix_seeds(1234)
    create_dirs(config.BASE_PATH)

    if args.train:
        train(args.epochs, args.n_train, args.k_train, args.q_train, n_eval=args.n_eval, k_eval=args.k_eval, q_eval=args.q_eval, episodes_per_epoch=args.episodes)
    else:
        parser.print_help()
import os
from argparse import ArgumentParser

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from models import protonet_embedding_model
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


def train(epochs, n_train, k_train, q_train, n_eval=1, k_eval=3, q_eval=5, episodes_per_epoch=100, num_tasks=1, lr=1e-3):
    # dataloaders for train and eval
    train_set = OmniglotDataset(subset='background')
    train_loader = DataLoader(train_set, num_workers=4, batch_sampler=FewShotBatchSampler(
        train_set, episodes_per_epoch=episodes_per_epoch, n=n_train, k=k_train, q=q_train, num_tasks=num_tasks
    ))
    eval_set = OmniglotDataset(subset='evaluation')
    eval_loader = DataLoader(eval_set, num_workers=4, batch_sampler=FewShotBatchSampler(
        eval_set, episodes_per_epoch=episodes_per_epoch, n=n_eval, k=k_eval, q=q_eval, num_tasks=num_tasks
    ))

    # train settings
    model = protonet_embedding_model()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.NLLLoss()  # .cuda()

    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, loss_fn, train_loader, n_train, k_train, q_train, epoch)
        evaluate(model, optimizer, loss_fn, train_loader, n_eval, k_eval, q_eval, epoch)


def train_epoch(model, optimizer, loss_fn, dataloader, n, k, q, epoch_idx):
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader, 1):
        # x.shape: [k * (n + q) * num_tasks, 1, 105, 105]
        # y.shape: [k * q]
        x, y = prepare_batch(batch, k, q)

        y_pred, loss = predict(x, y, model, loss_fn, n, k, q)

        loss.backward()
        optimizer.step()

        print(f'[epoch {epoch_idx} batch {i}] loss: {loss.item()}')


def evaluate(model, optimizer, loss_fn, dataloader, n, k, q, epoch_idx):
    model.eval()

    total_loss, data_cnt = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 1):
            x, y = prepare_batch(batch, k, q)
            
            y_pred, loss = predict(x, y, model, loss_fn, n, k, q)

            data_cnt += y_pred.shape[0]
            total_loss += loss.item() * y_pred.shape[0]

    print(f'[epoch {epoch_idx} eval] loss: {total_loss / data_cnt}')


def predict(x, y, model, loss_fn, n, k, q):
    embeddings = model(x)
    supports, queries = embeddings[:n * k], embeddings[n * k:]
    prototypes = supports.reshape(k, n, -1).mean(dim=1)

    distances = calc_distances(queries, prototypes, k, q)

    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    y_pred = (-distances).softmax(dim=1)

    return y_pred, loss


def prepare_batch(batch, k, q):
    # data.shape: [batch_size, channels, height, width] = [k * (n + q) * num_tasks, 1, 105, 105]
    # label.shape: [k * (n + q) * num_tasks] 
    # k * (n + q) means [n support sets of class 1 to k, q query sets of class 1 to k]
    data, label = batch

    x = data.float() # .cuda()

    # y = [*([0] * q), *([1] * q), ..., *([k - 1] * q)]
    # 正解ラベルはq個ごとにまとまっているため，q個ごとにclassification用のラベルを[0, k)で振り直す
    y = torch.arange(0, k, 1 / q).long() # .cuda()

    return x, y


def calc_distances(queries, prototypes, k, q):
    return (
        queries.unsqueeze(1).expand(k * q, k, -1) -
        prototypes.unsqueeze(0).expand(k * q, k, -1)
    ).pow(2).sum(dim=2)


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
    elif args.predict:
        predict()
    else:
        parser.print_help()
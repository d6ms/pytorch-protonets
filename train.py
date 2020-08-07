from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchsummary import summary

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker

import config
from models import protonet_embedding_model
from data import OmniglotDataset, FewShotBatchSampler

matplotlib.use('Agg')


def train(epochs, n_train, k_train, q_train, n_eval=1, k_eval=3, q_eval=5, episodes_per_epoch=100, num_tasks=1, lr=1e-3, lr_step_size=20, lr_gamma=0.5):
    # print parameters
    print('================ parameters ================')
    print('epochs', epochs)
    print('train (n, k, q)', n_train, k_train, q_train)
    print('eval (n, k, q)', n_eval, k_eval, q_eval)
    print('episodes per epoch', episodes_per_epoch)
    print('num_tasks', num_tasks)
    print('learning rate', lr)
    print('learning rate step size', lr_step_size)
    print('learning rate step rate', lr_gamma)
    print('============================================')

    # dataloaders for train and eval
    train_set = OmniglotDataset(subset='background')
    train_loader = DataLoader(train_set, num_workers=0, batch_sampler=FewShotBatchSampler(
        train_set, episodes_per_epoch=episodes_per_epoch, n=n_train, k=k_train, q=q_train, num_tasks=num_tasks
    ))
    eval_set = OmniglotDataset(subset='evaluation')
    eval_loader = DataLoader(eval_set, num_workers=0, batch_sampler=FewShotBatchSampler(
        eval_set, episodes_per_epoch=episodes_per_epoch, n=n_eval, k=k_eval, q=q_eval, num_tasks=num_tasks
    ))

    # train settings
    model = protonet_embedding_model().to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    loss_fn = torch.nn.NLLLoss().to(config.DEVICE)

    summary(model, (1, 105, 105))

    # train
    history = {'loss': list(), 'accuracy': list()}
    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, scheduler, loss_fn, train_loader, n_train, k_train, q_train, epoch)
        evaluate(model, history, loss_fn, eval_loader, n_eval, k_eval, q_eval, epoch)

        # save model and history
        if epoch == 1 or history['accuracy'][-1] > max(history['accuracy'][:-1]):
            torch.save(model.state_dict(), f'{config.MODEL_PATH}/protonets.ckpt')
        save_history(history)


def train_epoch(model, optimizer, scheduler, loss_fn, dataloader, n, k, q, epoch_idx):
    model.train()
    for i, batch in enumerate(dataloader, 1):
        optimizer.zero_grad()

        # x.shape: [k * (n + q) * num_tasks, 1, 105, 105]
        # y.shape: [k * q]
        x, y = prepare_batch(batch, k, q)

        y_pred, loss = predict(model, n, k, q, x, y, loss_fn)

        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'{datetime.now()} [epoch {epoch_idx} batch {i}] loss: {loss.item()}')


def evaluate(model, history, loss_fn, dataloader, n, k, q, epoch_idx):
    model.eval()

    total_loss, total_acc, data_cnt = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 1):
            x, y = prepare_batch(batch, k, q)
            
            y_pred, loss = predict(model, n, k, q, x, y, loss_fn)

            data_cnt += y_pred.shape[0]
            total_loss += loss.item() * y_pred.shape[0]
            total_acc += torch.eq(y_pred.argmax(dim=-1), y).sum().item()

    history['loss'].append(total_loss / data_cnt)
    history['accuracy'].append(total_acc / data_cnt)
    print(f'{datetime.now()} [epoch {epoch_idx} eval] loss: {total_loss / data_cnt}, accuracy: {total_acc / data_cnt}')


def predict(model, n, k, q, x, y=None, loss_fn=None):
    embeddings = model(x)
    supports, queries = embeddings[:n * k], embeddings[n * k:]
    prototypes = supports.reshape(k, n, -1).mean(dim=1)

    distances = calc_distances(queries, prototypes)

    log_p_y = (-distances).log_softmax(dim=1)
    if y is not None and loss_fn is not None:
        loss = loss_fn(log_p_y, y)
    else:
        loss = None

    y_pred = (-distances).softmax(dim=1)

    return y_pred, loss


def save_history(history):
    for metric, values in history.items():
        # save raw data
        with open(f'{config.LOG_PATH}/{metric}.data', mode='w') as f:
            data = ','.join([str(v) for v in values])
            f.write(data)
        # save graph
        x, y = np.linspace(1, len(values), len(values)), np.array(values)
        plt.figure()
        plt.plot(x, y, marker='o')
        plt.title(metric)
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.savefig(f'{config.LOG_PATH}/{metric}.png')


def prepare_batch(batch, k, q):
    # data.shape: [batch_size, channels, height, width] = [k * (n + q) * num_tasks, 1, 105, 105]
    # label.shape: [k * (n + q) * num_tasks] 
    # k * (n + q) means [n support sets of class 1 to k, q query sets of class 1 to k]
    data, label = batch

    x = data.float().to(config.DEVICE)

    # y = [*([0] * q), *([1] * q), ..., *([k - 1] * q)]
    # 正解ラベルはq個ごとにまとまっているため，q個ごとにclassification用のラベルを[0, k)で振り直す
    y = torch.arange(0, k, 1 / q).long().to(config.DEVICE)

    return x, y


def calc_distances(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


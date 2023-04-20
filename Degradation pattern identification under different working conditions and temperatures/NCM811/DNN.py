#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch import optim
import time
import warnings
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
from pathlib import Path
import os

warnings.filterwarnings('ignore')

class Dataset_generation(Dataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None


    def load_npy(self):
        self.x_train = np.array(pd.read_csv("train_preds.csv", header=None)).reshape(-1,51)
        self.x_valid = np.array(pd.read_csv("valid_preds.csv", header=None)).reshape(-1, 51)
        self.x_test = np.array(pd.read_csv("test_preds.csv", header=None)).reshape(-1, 51)
        self.y_train = np.array(pd.read_csv("train_trues.csv", header=None)).reshape(-1, 51)
        self.y_valid = np.array(pd.read_csv("valid_trues.csv", header=None)).reshape(-1, 51)
        self.y_test = np.array(pd.read_csv("test_trues.csv", header=None)).reshape(-1, 51)

    def return_dataloader(self):
        self.load_npy()
        print(self.x_train.shape,  self.y_train.shape,
              self.x_valid.shape,  self.y_valid.shape,
              self.x_test.shape,   self.y_test.shape)

        train_set = TensorDataset(torch.from_numpy(self.x_train).to(torch.float32),
                                  torch.from_numpy(self.y_train).to(torch.float32))
        valid_set = TensorDataset(torch.from_numpy(self.x_valid).to(torch.float32),
                                  torch.from_numpy(self.y_valid).to(torch.float32))
        test_set = TensorDataset(torch.from_numpy(self.x_test).to(torch.float32),
                                 torch.from_numpy(self.y_test).to(torch.float32))

        train_loader = DataLoader(train_set, batch_size=self.batch_size)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size)
        test_loader = DataLoader(test_set, batch_size=self.batch_size)

        return train_loader, valid_loader, test_loader



def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def adjust_learning_rate(optimizer, epoch, learning_rate, lradj):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.8 ** ((epoch - 1) // 2))}
    elif lradj == 'type2':
        lr_adjust = {
            100: 5e-3, 200: 1e-3, 300: 5e-4, 400: 1e-4,
            500: 5e-5, 600: 1e-5, 700: 5e-6, 800: 1e-6, 900: 5e-7
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.linear_layer1 = nn.Linear(51, 64)
        self.linear_layer2 = nn.Linear(64, 64)
        self.linear_layer3 = nn.Linear(64, 64)
        self.final_layer1 = nn.Linear(64, 51)
        self.final_layer2 = nn.Linear(64, 51)
        self.sigma_activation = nn.Softplus()
        self.mu_activation = nn.ELU()

    def forward(self, inputs):
        x = self.mu_activation(self.linear_layer1(inputs))
        x = self.mu_activation(self.linear_layer2(x))
        linear_out = self.linear_layer3(x)
        mu = self.final_layer1(linear_out)

        return mu



class Exp():
    def __init__(self,  patience):
        super(Exp, self).__init__()
        self.patience = patience
        self.learning_rate = 0.01
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MyModel().to(self.device)
        self.checkpoints_path = './checkpoint/DNN'

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_enc,  batch_output) in enumerate(vali_loader):
            batch_prediction, batch_output = self._process_one_batch(batch_enc, batch_output)
            loss = criterion(batch_prediction.detach().cpu(), batch_output.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, train_loader, vali_loader, test_loader, train_epochs):
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        my_file = Path(self.checkpoints_path + '/' + 'checkpoint.pth')
        if my_file.exists():
            best_model_path = self.checkpoints_path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            print('have loader best model')

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()
            for i, (batch_enc,  batch_output) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_prediction, batch_output = self._process_one_batch(batch_enc, batch_output)
                loss = criterion(batch_prediction, batch_output)
                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.checkpoints_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.learning_rate, lradj='type2')

        best_model_path = self.checkpoints_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def predict(self, test_loader, data_name):
        my_file = Path(self.checkpoints_path + '/' + 'checkpoint.pth')
        if my_file.exists():
            best_model_path = self.checkpoints_path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            print('have loader best model')
        self.model.eval()
        preds = []
        trues = []
        for i, (batch_enc, batch_output) in enumerate(test_loader):
            batch_prediction, batch_output = self._process_one_batch(batch_enc, batch_output)
            preds.append(batch_prediction.detach().cpu().numpy())
            trues.append(batch_output.detach().cpu().numpy())

        preds_last = preds.pop(-1)
        trues_last = trues.pop(-1)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, 51)
        preds = preds
        trues = trues.reshape(-1, 51)
        trues = trues

        preds_last = np.array(preds_last).reshape(-1, 51)
        trues_last = np.array(trues_last).reshape(-1, 51)

        preds = np.concatenate((preds, preds_last), axis=0)
        trues = np.concatenate((trues, trues_last), axis=0)

        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        pd.DataFrame(preds).to_csv(data_name + 'preds_second.csv', header=False, index=False)
        pd.DataFrame(trues).to_csv(data_name + 'trues_second.csv', header=False, index=False)

        return None

    def _process_one_batch(self, batch_enc, batch_output):
        batch_enc = batch_enc.float().to(self.device)
        batch_output = batch_output.float().to(self.device)
        batch_prediction = self.model(batch_enc)

        return batch_prediction, batch_output


data_generation = Dataset_generation(
    batch_size=1024)
train_loader, vali_loader, test_loader = data_generation.return_dataloader()

exp = Exp(patience=5000)
# print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
# exp.train(train_loader, vali_loader, test_loader, train_epochs=2400000)
# torch.cuda.empty_cache()
print('>>>>>>>start testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.predict(vali_loader, data_name="valid_")
exp.predict(test_loader, data_name="test_")
exp.predict(train_loader, data_name="train_")


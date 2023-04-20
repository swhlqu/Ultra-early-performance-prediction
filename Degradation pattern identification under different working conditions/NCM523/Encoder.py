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

        self.mean = None
        self.std = None

    def load_npy(self):
        self.x_train = np.load("x_train_523.npy").reshape(-1, 51, 2)
        self.x_valid = np.load("x_valid_523.npy").reshape(-1, 51, 2)
        self.x_test = np.load("x_test_523.npy").reshape(-1, 51, 2)
        self.y_train = np.load("y_train_523.npy").reshape(-1, 51)
        self.y_valid = np.load("y_valid_523.npy").reshape(-1, 51)
        self.y_test = np.load("y_test_523.npy").reshape(-1, 51)

    def norm_for_train(self):
        mean1 = np.mean(self.x_train[:, :, 0].reshape(-1, 1), axis=0).reshape(1, 1)
        mean2 = np.mean(self.x_train[:, :, 1].reshape(-1, 1), axis=0).reshape(1, 1)
        std1 = np.std(self.x_train[:, :, 0].reshape(-1, 1), axis=0).reshape(1, 1)
        std2 = np.std(self.x_train[:, :, 1].reshape(-1, 1), axis=0).reshape(1, 1)
        return mean1, mean2, std1, std2

    def norm_data(self, mean1, mean2, std1, std2):
        self.x_train[:, :, 0] = (self.x_train[:, :, 0] - mean1) / std1
        self.x_train[:, :, 1] = (self.x_train[:, :, 1] - mean2) / std2

        self.x_valid[:, :, 0] = (self.x_valid[:, :, 0] - mean1) / std1
        self.x_valid[:, :, 1] = (self.x_valid[:, :, 1] - mean2) / std2

        self.x_test[:, :, 0] = (self.x_test[:, :, 0] - mean1) / std1
        self.x_test[:, :, 1] = (self.x_test[:, :, 1] - mean2) / std2

    def return_dataloader(self):
        self.load_npy()
        print(self.x_train.shape, self.y_train.shape,
              self.x_valid.shape, self.y_valid.shape,
              self.x_test.shape, self.y_test.shape)
        mean1, mean2, std1, std2 = self.norm_for_train()
        self.norm_data(mean1, mean2, std1, std2)
        print(mean1, mean2, std1, std2)

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
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.8 ** ((epoch - 1) // 2))}
    elif lradj == 'type2':
        lr_adjust = {
            10: 5e-3, 20: 1e-3, 30: 5e-4, 40: 1e-4,
            50: 5e-5, 60: 1e-5, 70: 5e-6, 80: 1e-6, 90: 5e-7
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


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(
                math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads  # 8个头
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x = self.attention(
            x, x, x,
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x):
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x):
        x_stack = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
        x_stack = torch.cat(x_stack, -2)

        return x_stack


class Stack(nn.Module):
    def __init__(self, enc_in, c_out,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu',
                 distil=True,
                 device=torch.device('cuda:0')):
        super(Stack, self).__init__()

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout=0.05)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.flatten = nn.Flatten()

    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out)
        dec_out = self.projection(enc_out)
        dec_out = self.flatten(dec_out)

        return dec_out


class Exp():
    def __init__(self, enc_in, c_out, d_model, n_heads, e_layers,
                 d_ff, dropout, activation, distil, patience):
        super(Exp, self).__init__()
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.distil = distil
        self.patience = patience
        self.learning_rate = 0.01
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Stack(enc_in, c_out,
                           d_model, n_heads, e_layers, d_ff,
                           dropout, activation, distil).to(self.device)
        self.checkpoints_path = './checkpoint/Encoder'

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_enc, batch_output) in enumerate(vali_loader):
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
            for i, (batch_enc, batch_output) in enumerate(train_loader):
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

        pd.DataFrame(preds).to_csv(data_name + 'preds.csv', header=False, index=False)
        pd.DataFrame(trues).to_csv(data_name + 'trues.csv', header=False, index=False)

        return None

    def _process_one_batch(self, batch_enc, batch_output):
        batch_enc = batch_enc.float().to(self.device)
        batch_output = batch_output.float().to(self.device)
        batch_prediction = self.model(batch_enc)

        return batch_prediction, batch_output


data_generation = Dataset_generation(
    batch_size=1024)
train_loader, vali_loader, test_loader = data_generation.return_dataloader()

exp = Exp(enc_in=2, c_out=1,
          d_model=16, n_heads=4, e_layers=3, d_ff=16 * 4,
          dropout=0.05, activation='gelu',
          distil=True, patience=500)

# print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
# exp.train(train_loader, vali_loader, test_loader, train_epochs=2400000)
# torch.cuda.empty_cache()
print('>>>>>>>start testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.predict(vali_loader, data_name="valid_")
exp.predict(test_loader, data_name="test_")
exp.predict(train_loader, data_name="train_")

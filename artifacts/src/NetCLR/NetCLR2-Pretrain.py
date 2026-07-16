from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
warnings.filterwarnings('ignore')
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

import tqdm
import pickle
import argparse
import random
import math
import os
import bisect

import dill

from sklearn.utils import shuffle

# ==========================================
# 1. 设备配置与参数设置
# ==========================================

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
print(f"Device: {device}")

# 超参数
batch_size = 256
fp16_precision = True
temperature = 0.5
n_views = 2
num_epoches = 100

# ==========================================
# 2. 数据加载
# ==========================================

# 注意：请确保这个路径指向你的增强后的 npz 文件
data = np.load('../../datasets/AWF/awf1_aug2x.npz')

x_train = data['x_train']
y_train = data['y_train']

print(x_train.shape)
print(y_train.shape)

import numpy as np
np.set_printoptions(threshold=np.inf)
print(x_train[:1])

num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")

print(f'Train data shapes: {x_train.shape}, {y_train.shape}')

# ==========================================
# 3. 模型定义 (Backbone & Projection Head)
# ==========================================

class DFNet(nn.Module):
    def __init__(self, out_dim):
        super(DFNet, self).__init__()
        kernel_size = 8
        channels = [1, 32, 64, 128, 256]
        conv_stride = 1
        pool_stride = 4
        pool_size = 8
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size, stride=conv_stride)
        self.conv1_1 = nn.Conv1d(32, 32, kernel_size, stride=conv_stride)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size, stride=conv_stride)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size, stride=conv_stride)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size, stride=conv_stride)
        self.conv3_3 = nn.Conv1d(128, 128, kernel_size, stride=conv_stride)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size, stride=conv_stride)
        self.conv4_4 = nn.Conv1d(256, 256, kernel_size, stride=conv_stride)
        
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(256)
        
        self.max_pool_1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)
        
        self.fc = nn.Linear(5120, out_dim)

    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()
                
    def forward(self, inp):
        x = inp
        # ==== first block ====
        x = F.pad(x, (3,4))
        x = F.elu((self.conv1(x)))
        x = F.pad(x, (3,4))
        x = F.elu(self.batch_norm1(self.conv1_1(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_1(x)
        x = self.dropout1(x)
        
        # ==== second block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv2(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm2(self.conv2_2(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_2(x)
        x = self.dropout2(x)
        
        # ==== third block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv3(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm3(self.conv3_3(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_3(x)
        x = self.dropout3(x)
        
        # ==== fourth block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv4(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm4(self.conv4_4(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x 

class DFsimCLR(nn.Module):
    def __init__(self, df, out_dim):
        super(DFsimCLR, self).__init__()
        
        self.backbone = df
        self.backbone.weight_init()
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )
        
    def forward(self, inp):
        out = self.backbone(inp)
        return out

# ==========================================
# 4. 数据增强 (NetAugment)
# ==========================================

# def find_bursts(x):
#     direction = x[0]
#     bursts = []
#     start = 0
#     temp_burst = x[0]
#     for i in range(1, len(x)):
#         if x[i] == 0.0:
#             break
#         elif x[i] == direction:
#             temp_burst += x[i]
#         else:
#             bursts.append((start, i, temp_burst))
#             start = i
#             temp_burst = x[i]
#             direction *= -1
#     return bursts

# outgoing_burst_sizes = []
# x_random = x_train[np.random.choice(range(len(x_train)), size=1000, replace=False)]

# for x in x_random:
#     bursts = find_bursts(x)
#     outgoing_burst_sizes += [x[2] for x in bursts if x[2] > 0]

# max_outgoing_burst_size = max(outgoing_burst_sizes)

# bins = max(1, int(np.ceil(max_outgoing_burst_size - 1)))
# count, bins = np.histogram(outgoing_burst_sizes, bins=bins)
# PDF = count/np.sum(count)
# OUTGOING_BURST_SIZE_CDF = np.zeros_like(bins)
# OUTGOING_BURST_SIZE_CDF[1:] = np.cumsum(PDF)



# ==========================================
# 5. 数据加载器
# ==========================================
class TrainData(Dataset):
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train

    def __getitem__(self, index):
        idx = index * 2

        view1 = self.x[idx]
        view2 = self.x[idx + 1]

        label = self.y[idx]

        return [view1, view2], label

    def __len__(self):
        return len(self.x) // 2

# ==========================================
# 6. NetCLR 训练逻辑
# ==========================================

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class NetCLR(object):
    def __init__(self, **args):
        self.model = args['model']
        self.optimizer = args['optimizer']
        self.scheduler = args['scheduler']
        self.fp16_precision = args['fp16_precision']
        self.num_epoches = args['num_epoches']
        self.batch_size = args['batch_size']
        self.device = args['device']
        self.temperature = args['temperature']
        self.n_views = 2
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.log_every_n_step = 100
        
    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim = 0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        
        features = F.normalize(features, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T)
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        logits = logits / self.temperature
        return logits, labels
        
    def train(self, train_loader):
        best_acc = 0
        scaler = GradScaler(enabled=self.fp16_precision)

        n_iter = 0
        print ("Start SimCLR training for %d number of epoches"%self.num_epoches)
        
        first_loss = True
        for epoch_counter in range(self.num_epoches+1):
            with tqdm.tqdm(train_loader, unit='batch') as tepoch:
                for data, _ in tepoch:
                    tepoch.set_description(f"Epoch {epoch_counter}")
                    
                    self.model.train()
                    data = torch.cat(data, dim = 0)
                    data = data.view(data.size(0), 1, data.size(1))
                    data = data.float().to(self.device)

                    with autocast(enabled=self.fp16_precision):
                        features = self.model(data)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)

                    self.optimizer.zero_grad()
                    
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    if n_iter%self.log_every_n_step == 0:
                        top1, top5 = accuracy(logits, labels, topk=(1, 5))
                        tepoch.set_postfix(loss=loss.item(), accuracy = top1.item())
                    n_iter += 1

            if epoch_counter >= 10:
                self.scheduler.step()
            
            # saving the model each 
            if epoch_counter % 20 == 0:
                # 确保目录存在
                os.makedirs('./../models/NetCLR/', exist_ok=True)
                torch.save(self.model.state_dict(), f'./../models/NetCLR/NetCLR_epoch_{epoch_counter}.pth.tar')

# ==========================================
# 7. 主程序入口
# ==========================================

if __name__ == "__main__":
    temperature = 0.5 # this value is suggested by the original SimCLR paper
    
    train_dataset = TrainData(
        x_train,
        y_train
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    df = DFNet(out_dim=512)
    model = DFsimCLR(df, out_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003) #, weight_decay = 1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    netclr = NetCLR(model = model,
               optimizer = optimizer,
               scheduler = scheduler,
               fp16_precision = fp16_precision,
               device = device,
               temperature = temperature,
               n_views = n_views,
               num_epoches = 401,
               batch_size = batch_size)
    netclr.train(train_loader)
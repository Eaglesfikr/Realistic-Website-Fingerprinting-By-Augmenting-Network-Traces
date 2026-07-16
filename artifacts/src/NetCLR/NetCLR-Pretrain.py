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

# 注意：请确保这个路径指向你的 awf2.npz 文件
data = np.load('../../datasets/AWF/awf1.npz') # AWF-PT-sup
print(data.files)

x_train = data['data']
y_train = data['labels']

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

def find_bursts(x):
    direction = x[0]
    bursts = []
    start = 0
    temp_burst = x[0]
    for i in range(1, len(x)):
        if x[i] == 0.0:
            break
        elif x[i] == direction:
            temp_burst += x[i]
        else:
            bursts.append((start, i, temp_burst))
            start = i
            temp_burst = x[i]
            direction *= -1
    return bursts

outgoing_burst_sizes = []
x_random = x_train[np.random.choice(range(len(x_train)), size=1000, replace=False)]

for x in x_random:
    bursts = find_bursts(x)
    outgoing_burst_sizes += [x[2] for x in bursts if x[2] > 0]

max_outgoing_burst_size = max(outgoing_burst_sizes)

bins = max(1, int(np.ceil(max_outgoing_burst_size - 1)))
count, bins = np.histogram(outgoing_burst_sizes, bins=bins)
PDF = count/np.sum(count)
OUTGOING_BURST_SIZE_CDF = np.zeros_like(bins)
OUTGOING_BURST_SIZE_CDF[1:] = np.cumsum(PDF)

class Augmentor():
    def __init__(self):
        methods = {
            'merge downstream burst',
            'change downstream burst sizes',
            'merge downstream and upstream bursts',
            'add upstream bursts',
            'remove upstrean bursts',
            'divide bursts'
        }
        
        self.large_burst_threshold = 10
        
        # changing the content
        self.upsample_rate = 1.0
        self.downsample_rate = 0.5
        
        # merging bursts
        self.num_bursts_to_merge = 5
        self.merge_burst_rate = 0.1
        
        # add incoming bursts
        self.add_outgoing_burst_rate = 0.3
        self.outgoing_burst_sizes = list(range(max(1, int(np.floor(max_outgoing_burst_size)))))
        
        # shift
        self.shift_param = 10
        
    def find_bursts(self, x):
        direction = x[0]
        bursts = []
        start = 0
        temp_burst = x[0]
        for i in range(1, len(x)):
            if x[i] == 0.0:
                break
            elif x[i] == direction:
                temp_burst += x[i]
            else:
                bursts.append((start, i, temp_burst))
                start = i
                temp_burst = x[i]
                direction *= -1
        return bursts
    
    def increase_incoming_bursts(self, burst_sizes):
        out = []
        for i, size in enumerate(burst_sizes):
            if size <= -self.large_burst_threshold:
                up_sample_rate = random.random()*self.upsample_rate
                new_size = int(size * (1+up_sample_rate))
                out.append(new_size)
            else:
                out.append(size)
        return out
    
    def decrease_incoming_bursts(self, burst_sizes):
        out = []
        for i, size in enumerate(burst_sizes):
            if size <= -self.large_burst_threshold:
                up_sample_rate = random.random()*self.downsample_rate
                new_size = int(size * (1-up_sample_rate))
                out.append(new_size)
            else:
                out.append(size)
        return out
    
    def change_content(self, trace):
        bursts = self.find_bursts(trace)
        burst_sizes = [x[2] for x in bursts]
        
        if len(trace) < 1000:
            new_burst_sizes = self.increase_incoming_bursts(burst_sizes)
        elif len(trace) > 4000:
            new_burst_sizes = self.decrease_incoming_bursts(burst_sizes)
        else:
            p = random.random()
            if p >= 0.5:
                new_burst_sizes = self.increase_incoming_bursts(burst_sizes)
            else:
                new_burst_sizes = self.decrease_incoming_bursts(burst_sizes)
        return new_burst_sizes
    
    def merge_incoming_bursts(self, burst_sizes):
        out = []
        i = 0
        num_cells = 0
        while i < len(burst_sizes) and num_cells < 20:
            num_cells += abs(burst_sizes[i])
            out.append(burst_sizes[i])
            i += 1
        
        while i < len(burst_sizes) - self.num_bursts_to_merge:
            prob = random.random()
            if burst_sizes[i] > 0:
                out.append(burst_sizes[i])
                i+= 1
                continue
            
            if prob < self.merge_burst_rate:
                num_merges = random.randint(2, self.num_bursts_to_merge)
                merged_size = 0
                while i < len(burst_sizes) and num_merges > 0:
                    if burst_sizes[i] < 0:
                        merged_size += burst_sizes[i]
                        num_merges -= 1
                        i += 1 
                out.append(merged_size)
            else:
                out.append(burst_sizes[i])
                i += 1
        return out
    
    def add_outgoing_burst(self, burst_sizes):
        out = []
        i = 0
        num_cells = 0
        while i < len(burst_sizes) and num_cells < 20:
            num_cells += abs(burst_sizes[i])
            out.append(burst_sizes[i])
            i += 1
        
        for size in burst_sizes[i:]:
            if size > -10 :
                out.append(size)
                continue
            
            prob = random.random()
            if prob < self.add_outgoing_burst_rate:
                index = len(outgoing_burst_sizes)
                while index >= len(outgoing_burst_sizes):
                    outgoing_burst_prob = random.random()
                    index = bisect.bisect_left(OUTGOING_BURST_SIZE_CDF, outgoing_burst_prob)
                
                outgoing_burst_size = self.outgoing_burst_sizes[index]
                divide_place = random.randint(3, abs(size) - 3)
                out += [-divide_place, outgoing_burst_size, -(abs(size) - divide_place)]
            else:
                out.append(size)
        return out
    
    def create_trace_from_burst_sizes(self, burst_sizes):
        out = []
        for size in burst_sizes:
            val = 1 if size > 0 else -1
            out += [val]*(int(abs(size)))
        
        if len(out) < 5000:
            out += [0]*(5000 - len(out))
        return np.array(out)[:5000]
    
    def shift(self, x):
        pad = np.random.randint(0, 2, size = (self.shift_param, ))
        pad = 2*pad-1
        zpad = np.zeros_like(pad)
        
        shift_val = np.random.randint(-self.shift_param, self.shift_param+1, 1)[0]
        shifted = np.concatenate((x, zpad, pad), axis=-1)
        shifted = np.roll(shifted, shift_val, axis=-1)
        shifted = shifted[:5000]
        return shifted
    
    def augment(self, trace):
        mapping = {
            0: self.change_content,
            1: self.merge_incoming_bursts,
            2: self.add_outgoing_burst
        }
        
        bursts = self.find_bursts(trace)
        burst_sizes = [x[2] for x in bursts]
        
        aug_method = mapping[random.randint(0, len(mapping)-1)]
        augmented_sizes = aug_method(burst_sizes)
        augmented_trace = self.create_trace_from_burst_sizes(augmented_sizes)
        
        return self.shift(augmented_trace)

# ==========================================
# 5. 数据加载器
# ==========================================

class TrainData(Dataset):
    def __init__(self, x_train, y_train, augmentor, n_views):
        self.x = x_train
        self.y = y_train
        self.augmentor = augmentor
        self.n_views = n_views
        
    def _aug(self, inp):
        flip_idx = np.random.randint(0, 4999, 250)
        x_w = inp.copy()
        temp = x_w[flip_idx]
        x_w[flip_idx] = x_w[flip_idx+1]
        x_w[flip_idx+1] = temp
        return x_w
    
    def __getitem__(self, index):
        return [self.augmentor.augment(self.x[index]) for i in range(self.n_views)], self.y[index]
    
    def __len__(self):
        return len(self.x)

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

    augmentor = Augmentor()

    train_dataset = TrainData(x_train, y_train, augmentor, 2)
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
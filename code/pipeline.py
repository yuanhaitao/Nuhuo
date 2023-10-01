# -*- coding: utf-8 -*-
"""
    Author: yuanhaitao
    Date: 2022-03-26
"""
import enum
import copy
# import learn2learn as l2l
import collections
import torch.nn.functional as F
from torch.autograd import Variable
# from torch import threshold
# from multiprocessing import reduction
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import argparse
import os
import pandas as pd
import imp
import math
from numpy import array, zeros, argmin, inf, ndim
import matplotlib.pyplot as plt
from scipy import interpolate
from shapely.geometry import LineString
import json
import itertools
import re
from datetime import datetime
import time
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, _prepare_batch
from tqdm import tqdm
import torch.nn.functional as F
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
import glob
import yaml
import pickle
import random
import numpy as np
from datetime import timedelta
from ignite.handlers import EarlyStopping, ModelCheckpoint
from torch.optim import SGD, Adam, RMSprop
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, LineString, Point
import functools
import model
from model import SupConLoss
import data
from dgl.dataloading import GraphDataLoader
import dgl
from torch import autograd
import higher
import torch.optim as optim
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError(
        "No tensorboardX package is found. Please install with the command: \npip install tensorboardX")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
proj_dir = "/home_nfs/haitao/data/yht/DataCompletion/"
sys.path.append(proj_dir+"code/")

import model
import util
import gc
# import learn2learn as l2l

seed = 10
epsilon = 1e-6
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Process some integers.')
# source parameters
parser.add_argument('--city', type=str, required=True)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--training_file', type=str, required=True, help='the file of training dataset')
parser.add_argument('--validation_file',type=str, required=True, help='the file of val dataset')
parser.add_argument('--test_file',type=str, required=True, help='the file of test dataset')
parser.add_argument('--new_test_file',type=str, required=True, help='the file of test dataset')
parser.add_argument('--edge_file',type=str, required=True, help='the file of edge file')
parser.add_argument('--graph_file',type=str, required=True, help='the file of graph')
parser.add_argument('--test',type=int, default=0)
parser.add_argument('--resort_data',type=int, default=0)
# device="cpu"
# training_file="/home/hatim/data/DataCompletion/data/chengdu/train_output_dataset_0.2.txt"
# validation_file="/home/hatim/data/DataCompletion/data/chengdu/val_output_dataset_0.2.txt"
# test_file="/home/hatim/data/DataCompletion/data/chengdu/test_output_dataset_0.2.txt"
# edge_file="/home/hatim/data/DataCompletion/data/chengdu/new_edge_dict.pk"
# graph_file="/home/hatim/data/DataCompletion/data/chengdu/subgraphs_128.pk"

# training parameters
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=3)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_per_ep', type=int, default=500)
parser.add_argument('--target_replace_iter', type=int, default=50)
parser.add_argument('--topk', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--early_stop_epoch', type=int, default=20)
parser.add_argument('--grad_clamp', type=float, default=10)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--interval_size', type=int, default=15)
parser.add_argument('--data_ratio', type=float, default=1.0)

# model parameters
# in_dim = 2
# spatial_feature_dim = 2
parser.add_argument('--use_meta', type=int, default=1)
parser.add_argument('--use_sim', type=int, default=1)
parser.add_argument('--use_global', type=int, default=1)
parser.add_argument('--use_local', type=int, default=1)
parser.add_argument('--use_fusion', type=int, default=1)
parser.add_argument('--out_spatial_dim', type=int, default=100)
parser.add_argument('--out_temporal_dim', type=int, default=100)
parser.add_argument('--graph_layer', type=int, default=2)
parser.add_argument('--rnn_layer', type=int, default=2)
parser.add_argument('--spatial_context_dim', type=int, default=100)
parser.add_argument('--temporal_context_dim', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--threshold', type=float, default=0.1)
parser.add_argument('--rep_scale_weight',type=float, default=10.0)
parser.add_argument('--avg_speed_file', type=str, default=None)
parser.add_argument('--lambda_weight', type=float, default=0.1)

parser = parser.parse_args()
print(parser)



city = parser.city
device = parser.device
training_file=parser.training_file
validation_file=parser.validation_file
test_file=parser.test_file
new_test_file=parser.new_test_file
edge_file=parser.edge_file
graph_file=parser.graph_file
if device != "cpu" and torch.cuda.is_available():
    os.environ['CUDA_LAUNCH_BLOCKING'] = device[-1]
    torch.cuda.set_device(int(device[-1]))
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    device = torch.device(device)
else:
    device = torch.device("cpu")
train_batch_size=parser.train_batch_size
num_workers=parser.num_workers
test_batch_size=parser.test_batch_size
epochs=parser.epochs
batch_per_ep = parser.batch_per_ep
target_replace_iter = parser.target_replace_iter
topk = parser.topk
lr=parser.lr
early_stop_epoch=parser.early_stop_epoch
# lamb=parser.lamb
# alpha_1=parser.alpha_1
# alpha_2=parser.alpha_2
grad_clamp=parser.grad_clamp
weight_decay=parser.weight_decay

use_meta = True if  parser.use_meta>0 else False
use_sim = True if parser.use_sim>0 else False
use_global = True if parser.use_global>0 else False
use_local = True if parser.use_local>0 else False
use_fusion = True if parser.use_fusion>0 else False
test = True if parser.test>0 else False
resort_data = True if parser.resort_data>0 else False
out_spatial_dim = parser.out_spatial_dim
out_temporal_dim = parser.out_temporal_dim
graph_layer = parser.graph_layer
rnn_layer = parser.rnn_layer
spatial_context_dim = parser.spatial_context_dim
temporal_context_dim = parser.temporal_context_dim
# assert spatial_context_dim == temporal_context_dim
hidden_size = parser.hidden_size
threshold = parser.threshold
rep_scale_weight = parser.rep_scale_weight
lambda_weight = parser.lambda_weight
interval_size = parser.interval_size
data_ratio = parser.data_ratio

data.MyTrafficDataset.in_dim = 8
data.MyTrafficDatasetPath.in_dim = 8
data.MyTrafficDataset.time_slot_num = int(60//interval_size)
data.MyTrafficDatasetPath.time_slot_num = int(60//interval_size)


def my_collate_fn(batch):
    first_layer_idx = []
    weeks = []
    minutes = []
    # x = []
    mask = []
    sim_mask = []
    x = []
    y = []
    batch_g = []
    batch_idx = 0
    for data in batch:
        first_layer_idx.append(data['first_layer_idx'])
        weeks.append(data['weeks'])
        minutes.append(data['minutes'])
        mask += data['mask']
        sim_mask += data['sim_mask']
        x += data['x']
        y += data['y']
        g = data['graph']
        
        g.ndata['batch_idx'] = torch.LongTensor([batch_idx for _ in range(g.num_nodes())])
        batch_g.append(g)
        batch_idx += 1
    return torch.LongTensor(first_layer_idx), torch.LongTensor(weeks),\
        torch.LongTensor(minutes), dgl.batch(batch_g),\
            torch.FloatTensor(x), torch.FloatTensor(y), torch.LongTensor(mask), torch.LongTensor(sim_mask)
            # torch.softmax(torch.FloatTensor(x)+1.0/8, dim=-1), torch.softmax(torch.FloatTensor(y)+1.0/8, dim=-1), torch.LongTensor(mask), torch.LongTensor(sim_mask)

def train2(model, dataloader, val_dataloader, criterion, criterion_2, opt, opt_2, grad_clamp=1, use_meta=True, epoch=0):
    
    model.train()
    train_loss, rep_sim_loss = [], []
    # if use_sim:
    #             # 辅助任务先训练一轮
    #     for _ in tqdm(range(batch_per_ep)): 
    #         val_first_layer_idx, val_weeks, val_minutes, val_batch_g, val_raw_traffic, val_y, val_mask, val_sim_mask = next(val_dataloader)
    #         val_first_layer_idx, val_weeks, val_minutes, val_batch_g, val_raw_traffic, val_y, val_mask, val_sim_mask = \
    #                         val_first_layer_idx.to(device, non_blocking=True), val_weeks.to(device, non_blocking=True), \
    #                             val_minutes.to(device, non_blocking=True), val_batch_g.to(device), \
    #                                 val_raw_traffic.to(device, non_blocking=True), val_y.to(device, non_blocking=True), \
    #                                     val_mask.to(device, non_blocking=True), val_sim_mask.to(device, non_blocking=True)

    #         _, val_new_predict = model(x=(val_batch_g, (val_raw_traffic, val_y), val_weeks, val_minutes, val_first_layer_idx, val_mask, val_sim_mask),auxillary=True)
    #         minibatch_loss22 = criterion(val_new_predict, val_y, val_mask, val_sim_mask, reduction=True)
    #         opt_2.zero_grad()
    #         minibatch_loss22.backward()
    #         for param in model.second_round_parameters():
    #             if param.grad is not None:
    #                 param.grad.data.clamp_(-grad_clamp, grad_clamp)
    #         opt_2.step()      
                
    for _ in tqdm(range(batch_per_ep)): 
            data = next(dataloader)
            first_layer_idx, weeks, minutes, batch_g, raw_traffic, y, mask, sim_mask = data
            first_layer_idx, weeks, minutes, batch_g, raw_traffic, y, mask, sim_mask = \
                first_layer_idx.to(device, non_blocking=True), weeks.to(device, non_blocking=True), \
                    minutes.to(device, non_blocking=True), batch_g.to(device), raw_traffic.to(device, non_blocking=True), \
                        y.to(device, non_blocking=True), mask.to(device, non_blocking=True), sim_mask.to(device, non_blocking=True)

            _, new_predict = model(x=(batch_g, (raw_traffic, y), weeks, minutes, first_layer_idx, mask, sim_mask),auxillary=True)
            minibatch_loss2 = criterion(new_predict, y, mask, sim_mask, reduction=True)


            # if use_sim:
            #     # 辅助任务先训练一轮
            #     val_first_layer_idx, val_weeks, val_minutes, val_batch_g, val_raw_traffic, val_y, val_mask, val_sim_mask = next(val_dataloader)
            #     val_first_layer_idx, val_weeks, val_minutes, val_batch_g, val_raw_traffic, val_y, val_mask, val_sim_mask = \
            #                         val_first_layer_idx.to(device, non_blocking=True), val_weeks.to(device, non_blocking=True), \
            #                             val_minutes.to(device, non_blocking=True), val_batch_g.to(device), \
            #                                 val_raw_traffic.to(device, non_blocking=True), val_y.to(device, non_blocking=True), \
            #                                     val_mask.to(device, non_blocking=True), val_sim_mask.to(device, non_blocking=True)

            #     for _ in range(3):
            #         _, val_new_predict = model(x=(val_batch_g, (val_raw_traffic, val_y), val_weeks, val_minutes, val_first_layer_idx, val_mask, val_sim_mask),auxillary=True)
            #         minibatch_loss22 = criterion(val_new_predict, val_y, val_mask, val_sim_mask, reduction=True)
            #         opt_2.zero_grad()
            #         minibatch_loss22.backward()
            #         for param in model.second_round_parameters():
            #             if param.grad is not None:
            #                 param.grad.data.clamp_(-grad_clamp, grad_clamp)
            #         opt_2.step()
            

            if use_meta:
                # 和opt_2没有关系了
                for _ in range(2):
                    val_first_layer_idx, val_weeks, val_minutes, val_batch_g, val_raw_traffic, val_y, val_mask, val_sim_mask = next(val_dataloader)
                    val_first_layer_idx, val_weeks, val_minutes, val_batch_g, val_raw_traffic, val_y, val_mask, val_sim_mask = \
                                    val_first_layer_idx.to(device, non_blocking=True), val_weeks.to(device, non_blocking=True), \
                                        val_minutes.to(device, non_blocking=True), val_batch_g.to(device), \
                                            val_raw_traffic.to(device, non_blocking=True), val_y.to(device, non_blocking=True), \
                                                val_mask.to(device, non_blocking=True), val_sim_mask.to(device, non_blocking=True)
                    opt.zero_grad()
                    # inner_opt = optim.SGD(model.predict_parameters(), lr=lr/10.0)
                    with higher.innerloop_ctx(model, opt, copy_initial_weights=False) as (fnet, diffopt):
                        # _, meta_pre = fnet(x=(batch_g, y, weeks, minutes, first_layer_idx, mask, sim_mask), auxillary=True)
                        # meta_train_loss = criterion(meta_pre, y, mask, sim_mask, reduction=True, use_all=True)
        
                        # diffopt.step(meta_train_loss)
                        _, meta_predict = fnet(x=(val_batch_g, val_raw_traffic, val_weeks, val_minutes, val_first_layer_idx, val_mask, val_sim_mask),first_round=True)
                        meta_train_loss = criterion(meta_predict, val_y, val_mask, val_sim_mask, reduction=True)
                        diffopt.step(meta_train_loss)
                    
                # if use_sim:
                #     for _ in range(5):
                #         opt_2.zero_grad()
                #         _, meta_predict = model(x=(val_batch_g, val_y, val_weeks, val_minutes, val_first_layer_idx, val_mask, val_sim_mask),auxillary=True)
                #         minibatch_loss1 = criterion(meta_predict, val_y, val_mask, val_sim_mask, reduction=True)

                #         _, new_predict = model(x=(batch_g, y, weeks, minutes, first_layer_idx, mask, sim_mask),auxillary=True)
                #         minibatch_loss2 = criterion(new_predict, y, mask, sim_mask, reduction=True)

                #         (minibatch_loss2 + minibatch_loss1).backward()
                #         for param in model.second_round_parameters():
                #             if param.grad is not None:
                #                 param.grad.data.clamp_(-grad_clamp, grad_clamp)
                #         opt_2.step()
                # opt = opt_2



            # new_alpha_beta, _ =  model(x=(batch_g, (raw_traffic, y), weeks, minutes, first_layer_idx, mask, sim_mask),auxillary=True)

            # opt.zero_grad()
            # alpha_beta, predict = model(x=(batch_g, raw_traffic, weeks, minutes, first_layer_idx, mask, sim_mask),first_round=True)
            # minibatch_loss = criterion(predict, y, mask, sim_mask, reduction=True)
            
            opt.zero_grad()  
            alpha_beta1, alpha_beta2, predict1, predict2 = model(x=(batch_g, (raw_traffic, y), weeks, minutes, first_layer_idx, mask, sim_mask),sim=True)

            minibatch_loss = criterion(predict1, y, mask, sim_mask, reduction=True)
            minibatch_loss2 = criterion(predict2, y, mask, sim_mask, reduction=True)
            


            loss = minibatch_loss
            # sim_loss =  criterion_2(alpha_beta, new_alpha_beta.detach(), mask, sim_mask, reduction=True)
            sim_loss =  criterion_2(alpha_beta1, alpha_beta2, mask, sim_mask, reduction=True)
            if use_sim:
                    # # super_model = SupConLoss()
                    # # 随机采样10000个样本，如果不够的话就用全部的，然后构造样本的
                    # # # 通过mgda自适应地选择weight
                    # scales = {}
                    # grads = {}
                    # loss_data = {}
                    # # loss_list = [sim_loss1, sim_loss3]
                    # loss_list = [minibatch_loss, sim_loss]
                    # for pp, tt in enumerate(loss_list):
                    #     opt.zero_grad()
                    #     grads[pp] = []
                    #     loss_data[pp] = tt.item()
                    #     tt.backward(retain_graph=True)
                    #     for param in model.parameters():
                    #         if param.grad is not None:
                    #             grads[pp].append(Variable(param.grad.data.clone(), requires_grad = False))
                    # gn = util.gradient_normalizers(grads, loss_data, "l2")
                    # # print(gn)
                    # for t, g in grads.items():
                    #     for gr_i in range(len(g)):
                    #         grads[t][gr_i] = grads[t][gr_i] / gn[t]

                    # # print(grads)
                    # try:
                    #     sol, min_norm = util.MinNormSolver.find_min_norm_element([grads[t] for t in grads.keys()])
                    # except Exception as err:
                    #     sol, min_norm  = None, None
                    # for t, _ in grads.items():
                    #     scales[t]  = float(sol[t]) if sol is not None else None
                    # opt.zero_grad()
                    # # loss = 0.0
                    # for i in range(len(loss_list)):
                    #     sl = scales[i]
                    #     if sl is None or np.isnan(sl) or np.isinf(sl):
                    #         sl = 1.0
                    #     if i == 0:
                    #         loss = loss_list[i] * sl
                    #     else:
                    #         loss = loss + loss_list[i] * sl

                    # loss = loss + sim_loss + minibatch_loss2
                    loss = loss + lambda_weight * (sim_loss + minibatch_loss2)
            
            # opt_2.zero_grad()
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clamp, grad_clamp)
            # opt2.step()
            opt.step()


            train_loss.append(loss.cpu().detach().numpy())
            rep_sim_loss.append([minibatch_loss.cpu().detach().numpy(),
            sim_loss.cpu().detach().numpy()]) #,sim_loss3.cpu().detach().numpy()])
    
            
    return np.mean(train_loss), np.mean(np.array(rep_sim_loss),axis=0)


#L2 正则化
def L2Loss(model,alpha):
    l2_loss = torch.tensor(0.0,requires_grad = True)
    for name,parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma,2)))
    return l2_loss

def L1Loss(model,beta):
    l1_loss = torch.tensor(0.0,requires_grad = True)
    for name,parma in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(parma))
    return l1_loss

# def train(model, criterion, opt, loader, grad_clamp=1):
#     model.train()
#     train_loss = []
#     # valid_value_num = 0
#     for _ in tqdm(range(batch_per_ep)):
#         # print("data size:", len(data))
#         # try:
#         data = next(loader)
#         first_layer_idx, weeks, minutes, batch_g, x, y, mask, sim_mask = data
#         first_layer_idx, weeks, minutes, batch_g, x, y, mask, sim_mask = \
#             first_layer_idx.to(device, non_blocking=True), weeks.to(device, non_blocking=True), \
#                 minutes.to(device, non_blocking=True), batch_g.to(device), \
#                     x.to(device, non_blocking=True), y.to(device, non_blocking=True), \
#                         mask.to(device, non_blocking=True), sim_mask.to(device, non_blocking=True)

#         opt.zero_grad()
#         batch_g.ndata['traffic'] = x
#         batch_g.ndata['mask'] = mask.float()

#         predict = model(x=(batch_g, x, weeks, minutes, first_layer_idx, mask, sim_mask),first_round=True)[-1]
#         minibatch_loss = criterion(predict, y, mask, sim_mask, reduction=True)
#         if minibatch_loss is None:
#             continue
#         # minibatch_loss = torch.mean(minibatch_loss)
#         # l2_loss = L2Loss(model,0.001)
#         # l1_loss = L1Loss(model,0.001)
#         total_loss =  minibatch_loss
#         # minibatch_loss.backward()
#         total_loss.backward()
#         for param in model.parameters():
#             if param.grad is not None:
#                 param.grad.data.clamp_(-grad_clamp, grad_clamp)
#         opt.step()
#         train_loss.append(minibatch_loss.cpu().detach().numpy())
#         torch.cuda.empty_cache()
#     if len(train_loss) == 0:
#         return None
#     return np.mean(train_loss)


import torchmetrics
def validation(model, dataloader, avg_speed_tensor, resort_data = False):
    model.eval()
    import data as datap
    kl_metric_ha = nn.KLDivLoss(reduction="batchmean").to(device)
    kl_metric_pre = nn.KLDivLoss(reduction="batchmean").to(device)
    mse_metric_pre_detail = nn.MSELoss(reduction="none").to(device)
    mse_metric_pre = nn.MSELoss(reduction="mean").to(device)
    ha_loss, pre_loss1, mse_loss1, pre_loss2, mse_loss2 =  [], [], [], [], []
    # pre_loss_detail = []
    with torch.no_grad():
        # total_num, delta_num = 0, 0
        # for data in tqdm(dataloader):
        # batch_loss_record = []
        for _ in tqdm(range(batch_per_ep if resort_data is False else 100000000)):
            try:
                data = next(dataloader)
            except StopIteration:
                break
            first_layer_idx, weeks, minutes, batch_g, x, y, mask, sim_mask = data
            first_layer_idx, weeks, minutes, batch_g, x, y, mask, sim_mask = \
            first_layer_idx.to(device, non_blocking=True), weeks.to(device, non_blocking=True), \
                minutes.to(device, non_blocking=True), batch_g.to(device), \
                    x.to(device, non_blocking=True), y.to(device, non_blocking=True), \
                        mask.to(device, non_blocking=True), sim_mask.to(device, non_blocking=True)
            _, predict_1 = model(x=(batch_g, x, weeks, minutes, first_layer_idx, mask, sim_mask),first_round=True)
            # _, predict_2 = model(x=predict_1,first_round=False)
            _, predict_2 = model(x=(batch_g, predict_1, weeks, minutes, first_layer_idx, mask, sim_mask),first_round=True)
            assert torch.isnan(predict_1).any() == False and torch.isinf(predict_1).any() == False
            assert torch.isnan(predict_2).any() == False and torch.isinf(predict_2).any() == False
            assert torch.isnan(y).any() == False and torch.isinf(y).any() == False

            # batch_size = predict_1.shape[0]

            # tmp = list(mse_metric_pre_detail(predict_1.reshape(batch_size, -1), y.reshape(batch_size, -1)).cpu().detach().numpy())
            # if len(pre_loss_detail) == 0:
            #     print("tmp", len(tmp))
            # pre_loss_detail += tmp

            mask = mask.unsqueeze(2).repeat(1,1,datap.MyTrafficDataset.in_dim)
            sim_mask = sim_mask.unsqueeze(2).repeat(1,1,datap.MyTrafficDataset.in_dim)
            
            y = torch.masked_select(y, mask > 0).reshape([-1, datap.MyTrafficDataset.in_dim])
            y_pre1 = torch.masked_select(predict_1, mask > 0).reshape([-1, datap.MyTrafficDataset.in_dim])
            y_pre2 = torch.masked_select(predict_2, mask > 0).reshape([-1, datap.MyTrafficDataset.in_dim])
            
            #return
            # y, y_pre1, y_pre2 = torch.log(y), torch.log(y_pre1), torch.log(y_pre2)
            
            y_ha = avg_speed_tensor.index_select(0, batch_g.ndata['spatial_idx']).unsqueeze(1).repeat(1, x.shape[1], 1)
            y_ha = torch.masked_select(y_ha, mask > 0).reshape([-1, datap.MyTrafficDataset.in_dim])
            
            y_ha = (y_ha + epsilon)/torch.sum(y_ha + epsilon, dim=-1, keepdim=True)
            y = (y + epsilon)/torch.sum(y + epsilon, dim=-1, keepdim=True)
            y_pre1 = (y_pre1 + epsilon)/torch.sum(y_pre1 + epsilon, dim=-1, keepdim=True)
            y_pre2 = (y_pre2 + epsilon)/torch.sum(y_pre2 + epsilon, dim=-1, keepdim=True)
            
            if y_ha.numel() == 0 or y.numel() == 0 or y_pre1.numel() == 0 or y_pre2.numel() == 0:
                print("the number of test is 0")
                continue
            # pre_loss_detail += list(kl_metric_pre_detail(y_pre1.log(), y).cpu().detach().numpy())
            ha_loss.append(kl_metric_ha(y_ha.log(), y).cpu().detach().numpy())
            pre_loss1.append(kl_metric_pre(y_pre1.log(), y).cpu().detach().numpy())
            mse_loss1.append(mse_metric_pre(y_pre1, y).cpu().detach().numpy())
            pre_loss2.append(kl_metric_pre(y_pre2.log(), y).cpu().detach().numpy())
            mse_loss2.append(mse_metric_pre(y_pre2, y).cpu().detach().numpy())
            torch.cuda.empty_cache()
    if len(pre_loss1) == 0 or len(ha_loss) == 0 or len(mse_loss1) == 0\
        or len(pre_loss2) == 0 or len(mse_loss2) == 0:
        return None
    return mse_loss1, (np.mean(pre_loss1)/np.mean(ha_loss), np.mean(mse_loss1)), \
        (np.mean(pre_loss2)/np.mean(ha_loss), np.mean(mse_loss2))
    # else:
    #     return (np.mean(pre_loss1)/np.mean(ha_loss), np.mean(mse_loss1)), \
    #     (np.mean(pre_loss2)/np.mean(ha_loss), np.mean(mse_loss2))
            
            

def traffic_loss(input, target, mask, sim_mask, reduction=False, only_sim = False, use_all=False):
    import data as datap
    mask = mask.unsqueeze(2).repeat(1,1,datap.MyTrafficDataset.in_dim)
    sim_mask = sim_mask.unsqueeze(2).repeat(1,1,datap.MyTrafficDataset.in_dim)
    if use_all:
        input = torch.masked_select(input, mask >= 0).reshape([-1, datap.MyTrafficDataset.in_dim])
        target = torch.masked_select(target, mask >= 0).reshape([-1, datap.MyTrafficDataset.in_dim])
    elif only_sim:
        input = torch.masked_select(input, sim_mask > 0).reshape([-1, datap.MyTrafficDataset.in_dim])
        target = torch.masked_select(target, sim_mask > 0).reshape([-1, datap.MyTrafficDataset.in_dim])
    else:
        input = torch.masked_select(input, mask > 0).reshape([-1, datap.MyTrafficDataset.in_dim])
        target = torch.masked_select(target, mask > 0).reshape([-1, datap.MyTrafficDataset.in_dim])
    
    if input.numel() == 0 or target.numel() == 0:
        return None
    
    
    input = (input + epsilon)/torch.sum(input + epsilon, dim=-1, keepdim=True)
    target = (target + epsilon)/torch.sum(target + epsilon, dim=-1, keepdim=True)

    if reduction:
        loss_function = nn.KLDivLoss(reduction="batchmean").to(device)
        loss = loss_function(input.log(), target)
    else:
        loss_function = nn.KLDivLoss(reduction="none").to(device)
        loss = loss_function(input.log(), target)
        loss = torch.sum(loss, dim=-1)
    return loss
    

def get_topk(data, descending=True):
    if  data.size(0) <= topk:
        # 如果数据不足topk，那么全返回
        return data
    _, idx1 = torch.sort(data, descending=descending)
    _, idx2 = torch.sort(idx1)
    return data[idx2<topk]
    
def rep_loss(rep, mask, sim_mask):
    loss_function = nn.CrossEntropyLoss(reduction = 'none')
    
    pre_y = torch.masked_select(rep, mask.unsqueeze(2).repeat(1,1,2) >= 0).view(-1, 2)
    # 一共两种
    target_y = torch.masked_select(mask, mask >= 0)
    
    #计算loss
    return loss_function(pre_y, target_y)

def rep_loss_2(rep1, rep2, mask, sim_mask, reduction=False):
    import data as datap
    mask = mask.unsqueeze(2).repeat(1,1,rep1.shape[-1])
    sim_mask = sim_mask.unsqueeze(2).repeat(1,1,rep1.shape[-1])
    input = torch.masked_select(rep1, mask > 0).reshape([-1, rep1.shape[-1]])
    target = torch.masked_select(rep2, mask > 0).reshape([-1, rep1.shape[-1]])
    if reduction:
        loss_function = nn.MSELoss(reduction = 'mean')
    else:
        loss_function = nn.MSELoss(reduction = 'none')
    return loss_function(input, target)
    



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    avg_speed_file = parser.avg_speed_file
    avg_speed_tensor = None
    if avg_speed_file is not None:
        abc = pickle.load(open(edge_file,"rb"))
        edge_size = max(list(abc[1].keys())) + 1
        assert edge_size > 0
        avg_speed_list = [None for _ in range(edge_size)]
        hist_size = None
        for line in open(avg_speed_file, "r"):
            line = line.strip().split("\t")
            lid = int(line[0])
            avg_speed_list[lid] = [float(t) for t in line[1:]]
            if hist_size is None:
                hist_size = len(line) - 1
        # check 所有none的
        for i, v in enumerate(avg_speed_list):
            if v is None:
                avg_speed_list[i] = [round(1.0/hist_size, 2) for _ in range(hist_size)]
        avg_speed_tensor = torch.FloatTensor(avg_speed_list).to(device)
        # avg_speed_tensor = torch.exp(avg_speed_tensor)
        # avg_speed_tensor = torch.softmax(avg_speed_tensor+1.0/8, dim=-1)
    
    train_dataset = data.MyTrafficDatasetPath(
        file_path=training_file,
        edge_file=edge_file,
        graph_file=graph_file,
        ratio = data_ratio
    )

    val_dataset = data.MyTrafficDataset(
        file_path=validation_file,
        edge_file=edge_file,
        graph_file=graph_file
    )
    
    val_dataset_unlimit = data.MyTrafficDataset(
        file_path=validation_file,
        edge_file=edge_file,
        graph_file=graph_file,
        unlimit = True
    )

    
    train_dataloader = iter(DataLoader(train_dataset, collate_fn= my_collate_fn, \
        shuffle=False, batch_size=train_batch_size, num_workers=num_workers))
    # val_dataloader = DataLoader(val_dataset, collate_fn= my_collate_fn, \
    #     shuffle=False, batch_size=test_batch_size, num_workers=num_workers)
    # meta_dataloader = itertools.cycle(DataLoader(train_dataset, collate_fn= my_collate_fn, \
    #     shuffle=False, batch_size=train_batch_size))

    # 构造model
    in_dim, spatial_feature_dim = data.MyTrafficDataset.in_dim, data.MyTrafficDataset.spatial_feature_dim
    time_slot_num = data.MyTrafficDataset.time_slot_num
    region_nums, link_nums = len(train_dataset.first_layer_edges), \
        len(train_dataset.new_part2id_dict)
    region_edge_start = []
    region_edge_end = []
    for i, p in enumerate(train_dataset.first_layer_edges):
        region_edge_start += [i for _ in p]
        region_edge_end += p
    region_edges =(region_edge_start, region_edge_end)
    meta_model = model.MetaSTC(time_slot_num, in_dim, spatial_feature_dim,
                    out_spatial_dim, out_temporal_dim, graph_layer, 
                    rnn_layer, spatial_context_dim, temporal_context_dim,
                    region_nums, link_nums, region_edges, hidden_size, avg_speed_tensor=avg_speed_tensor, device=device,
                    use_global = use_global, use_local = use_local, use_fusion = use_fusion)
    meta_model.to(device)
    
    
    # meta_target_model = model.MetaSTC(in_dim, spatial_feature_dim,
    #                 out_spatial_dim, out_temporal_dim, graph_layer, 
    #                 rnn_layer, spatial_context_dim, temporal_context_dim,
    #                 region_nums, link_nums, region_edges, hidden_size, avg_speed_tensor=avg_speed_tensor, device=device)
    # meta_target_model.to(device)
    # meta_model.share_memory()
    # mp.set_start_method('spawn')

    # if use_sim:
    #     opt = optim.Adam(meta_model.first_round_parameters(), lr=lr, weight_decay=weight_decay)
    # else:
    opt = optim.Adam(meta_model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_2 = optim.Adam(meta_model.second_round_parameters(), lr=lr, weight_decay=weight_decay)
    # opt_2 = optim.Adam(meta_model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = None
    last_best_ep = 0
    writer = SummaryWriter('{}/log'.format(proj_dir))
    time_flag = str(int(time.time())-1600000000)
    
    if test:
        # if resort_data is False:
        #     test_file = new_test_file
        #     num_workers = 0
        print("the test file is:{}".format(test_file))
        test_dataset = data.MyTrafficDataset(
            file_path=test_file,
            edge_file=edge_file,
            graph_file=graph_file
        )
        test_dataloader = iter(DataLoader(test_dataset, collate_fn= my_collate_fn, \
            shuffle=False, batch_size=test_batch_size, num_workers=num_workers, drop_last = True))
        
        model_name = "{}/model/{}_Meta_model_{}_{}_{}_{}_{}_{}_".format(proj_dir,city, use_meta, use_sim, use_global, use_local, use_fusion, interval_size)
        files = glob.glob(model_name+"*")
    #     print(file_pre,files)
        files_suf = [list(map(int, f[len(model_name):f.index(".pt")].split("_"))) for f in files]
        
        start_epoch = 0
        if len(files)>0:
            load_f = files[np.argmax([sum(t) for t in files_suf])]
            print("the model file is: ", load_f)
            meta_model.load_state_dict(torch.load(load_f))
        
        # r
        pre_loss1, epoch_loss_val_1, epoch_loss_val_2= validation(meta_model, test_dataloader, avg_speed_tensor, resort_data)
        
        # if resort_data:
        #     test_dataset._resort_data_and_rewrite(pre_loss1, test_batch_size, new_test_file)
            
        print("Validation_MKLR_Loss_1r: {},  Validation_MSE_Loss_1r: {}".format(epoch_loss_val_1[0], epoch_loss_val_1[1]))
        print("Validation_MKLR_Loss_2r: {},  Validation_MSE_Loss_2r: {}".format(epoch_loss_val_2[0], epoch_loss_val_2[1]))
        
    else:
        for ep in range(1,epochs+1):
                
                val_dataloader_unlimit = iter(DataLoader(val_dataset_unlimit, collate_fn= my_collate_fn, \
            shuffle=False, batch_size=test_batch_size, num_workers=num_workers))
                
                # epoch_loss, epoch_sim_loss = train2(meta_model, train_dataloader, val_dataloader_unlimit, traffic_loss, rep_loss_2, opt, opt_2, grad_clamp=grad_clamp, use_meta=use_meta, epoch=ep)
                # epoch_loss = train(meta_model, traffic_loss, opt, train_dataloader, grad_clamp=grad_clamp)
                
                val_dataloader = iter(DataLoader(val_dataset, collate_fn= my_collate_fn, \
            shuffle=False, batch_size=test_batch_size, num_workers=num_workers))
                _, epoch_loss_val_1, epoch_loss_val_2= validation(meta_model, val_dataloader, avg_speed_tensor)
                break
                
                key = "{}_{}_{}_{}_{}_{}_{}_{}".format(use_meta, use_sim, use_global, use_local, use_fusion, interval_size, data_ratio, time_flag)
                writer.add_scalars('{}_Meta/Train_Loss'.format(city), {key : epoch_loss}, ep)
                writer.add_scalars('{}_Meta/Train_Minibatch_Loss'.format(city), {key :epoch_sim_loss[0]}, ep)
                writer.add_scalars('{}_Meta/Train_Sim_Loss'.format(city), {key :epoch_sim_loss[1]}, ep)
                writer.add_scalars('{}_Meta/Validation_MKLR_Loss_1r'.format(city), {key :epoch_loss_val_1[0]}, ep)
                writer.add_scalars('{}_Meta/Validation_MSE_Loss_1r'.format(city), {key :epoch_loss_val_1[1]}, ep)
                writer.add_scalars('{}_Meta/Validation_MKLR_Loss_2r'.format(city), {key :epoch_loss_val_2[0]}, ep)
                writer.add_scalars('{}_Meta/Validation_MSE_Loss_2r'.format(city), {key :epoch_loss_val_2[1]}, ep)
                # writer.flush()
                if best_loss is None or epoch_loss_val_1[0] < best_loss or epoch_loss_val_2[0] < best_loss:
                    best_loss = min(epoch_loss_val_1[0], epoch_loss_val_2[0])
                    model_name = "{}/model/{}_Meta_model_{}_{}.pt".format(proj_dir,city, key,ep)
                    torch.save(meta_model.state_dict(), model_name)
                    last_best_ep = ep
                if ep - last_best_ep > early_stop_epoch:
                    break
            
    
       
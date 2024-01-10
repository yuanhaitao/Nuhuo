# -*- encoding:utf-8 -*-
# 构造样本，每条样本是采样的一个子图（空间上），并且每个时间戳考虑其前后各一个小时（时间上）

#2.加载link分层图+link属性信息
import os
# import metis
import pymetis as metis
import pandas as pd
from geopy.distance import geodesic
import pickle
import queue
import sys
raw_data_path="/home/hatim/data/EventDetection/chengdu"
my_data_path="/home/hatim/data/DataCompletion/data/chengdu"
    
import pickle
old2new_edge_dict, new2old_edge_dict, new_edge2edge_neighbor_dict = pickle.load(open(my_data_path+"/new_edge_dict.pk","rb"))
f = open(my_data_path+"/subgraphs_128.pk","rb")
first_layer_edges, second_layer_edges, new_id2part_dict, nxt_list = pickle.load(f)
new_part2id_dict = {(v[0],v[1]):k for k,v in new_id2part_dict.items()}

def load_traffic(files):
    '''加载所有的traffic信息'''
    res = {}
    for file in files:
        for log in open(file):
            data = log.split("\t")
            data = [d.strip() for d in data]
            assert len(data) == 290
            linkid, day = data[:2]
            linkid = int(linkid)
            tmp = res.get(linkid,{})
            tmp[day] = data[2:]
            res[linkid] = tmp
    return res

import datetime
import copy
# begin = datetime.date(2016,10,1)
# end = datetime.date(2016,11,30)
# end = datetime.date(2016,10,7)
def build_dataset(begin, end, time_slot_num = 12, slip_step = 1):
    # 1.先加载数据
    d = begin
    delta = datetime.timedelta(days=1)
    days = []
    while d <= end:
        date = d.strftime("%Y%m%d")
        days.append(date)
        d += delta
    traffic_files = [my_data_path+"/traffic_{}_detail".format(day) for day in days]
    traffic_dict = load_traffic(traffic_files)
    #2.利用加载的link分层图+link属性信息进行样本的构建（首先是link子图的遍历，其次是时间的遍历）
    first_num = len(first_layer_edges)
    for first_i in range(first_num):
        s_edges = second_layer_edges[first_i]
        second_num = len(s_edges)
        # 该层所有节点构成一个空间样本，接下来是时间遍历（根据begin和end确定的时间区间，如果是空的，那么就用''补全）
        # train_data = [] #按照时间片（一小时时间片，平移，slip的步数有参数确定,默认是1）
        last_date = None
        last_slot_info = []
        d = begin
        while d <= end:
            date = d.strftime("%Y%m%d")
            # 获取全天的子图列表（如果是空的，就用‘’补齐）
            data_list = []
            for second_i in range(second_num):
                data_list.append(traffic_dict.get(new_part2id_dict[(first_i, second_i)],{}).get(date,['' for _ in range(288)]))
            # print(date,len(data_list))
            if last_date != None:
                #那么last_data的最后time_slot_num-1个时间片信息可以接到今天
                for _t in range(0,time_slot_num-1,slip_step):
                    pre_set = [info[_t:] for info in last_slot_info]
                    assert len(pre_set) == len(data_list)
                    nxt_set = [info[:_t+1] for info in data_list]
                    train_set = [pre+nxt for pre,nxt in zip(pre_set, nxt_set)]
                    train_set = ['|'.join([train_set[s_i][t_i] for s_i in range(second_num)])  for t_i in range(time_slot_num)]
                    train_set = '\t'.join([str(first_i), last_date, str(288-(time_slot_num-_t)+1)] + train_set)
                    print(train_set)
            for _t in range(0,288-time_slot_num+1,slip_step):
                train_set = ['|'.join([data_list[s_i][_t+t_i] for s_i in range(second_num)])  for t_i in range(time_slot_num)]
                train_set = '\t'.join([str(first_i), date, str(_t)] + train_set)
                print(train_set)
            last_date = date
            last_slot_info = [copy.deepcopy(data[288-time_slot_num+1:]) for data in data_list]
            d += delta

if __name__ == '__main__':
    # 通过参数
    begin_month = int(sys.argv[1])
    begin_day = int(sys.argv[2])
    end_month = int(sys.argv[3])
    end_day = int(sys.argv[4])
    begin = datetime.date(2016,begin_month,begin_day)
    end = datetime.date(2016,end_month,end_day)
    build_dataset(begin, end)
    
    
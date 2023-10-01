# -*- encoding: utf-8 -*-
from fileinput import filename
# from sklearn.preprocessing import label_binarize
# from torch.utils.data import Dataset
import tqdm
import torch
import random
import pickle
import datetime
import dgl
import os
# random.seed(1000)
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import itertools

class TrafficDatasetDetail2():
    def _generate_(self, data_path, edge_file, graph_file, output_path,
                 time_interval_size = 15,
                 valid_min_size = 5, 
                 mask_rate = 0.2,
                 hist_size = 8,
                 max_speed = 40,
                 max_data_size=None):
        '''
            data_path: the file of data
            time_interval_size: how many nimutes each time interval includes
            valid_min_size: the minimum size of valid time slots, if the real size is less than it, set the value with zero
            mask_rate: how many samples are set with MASK, and the value is replaced with zero
            hist_size: the number of hists
            max_speed: the maximum value for the speed
            max_data_size: for limiting the number of data
        '''
        self.merge_interval = int(time_interval_size // 5)
        assert self.merge_interval >= 1
        self.valid_min_size = valid_min_size
        self.mask_rate = mask_rate
        self.hist_size = hist_size
        self.delta_speed = int(40/hist_size)
        self.max_speed = max_speed
        # self.lines = []
        _, self.new2old_edge_dict, _ = pickle.load(open(edge_file,"rb"))
        _, self.second_layer_edges, new_id2part_dict, _ = pickle.load(open(graph_file,"rb"))
        self.new_part2id_dict = {(v[0],v[1]):k for k,v in new_id2part_dict.items()}
        self.speedlimit_dict = {'bridleway':10, 'construction':0, 'cycleway':15, 'footway':5, 'living_street':5,
        'motorway':50,'motorway_link':50,'path':5,'pedestrian':5,'platform':5,'primary':40,'primary_link':40,
        'raceway':30,'residential':5,'road':10,'secondary':30,'secondary_link':30,'service':10,'steps':5,
        'tertiary':10,'tertiary_link':10,'track':5,'trunk':50,'trunk_link':50,'unclassified':20,'subway':50,'rail':40}
        random.seed(os.path.getsize(data_path))
        cum_num = 0
        with open(output_path, "w") as writer:
            for line in open(data_path):
                data = line.strip().split("\t")
                first_layer_idx = int(data[0])
                day = int(data[1])
                start_time_slot = int(data[2]) // self.merge_interval
                batch_data = [] 
                raw_batch_data = [t.split("|") for t in data[3:]]
                for l in range(len(raw_batch_data[0])):
                    seq = [raw_batch_data[t][l] for t in range(len(raw_batch_data))]
                    new_seq = []
                    for s in seq:
                        if s == '':
                            new_seq.append(None)
                        else:
                            # speed, traj_num = s.split(",")
                            new_seq.append([float(speed) for speed in s.split(",")])
                    batch_data.append(new_seq)
                input_data, output_data, label_mask, label_sim_mask = self.render_data(first_layer_idx, batch_data)
                # self.lines.append((first_layer_idx, day, start_time_slot, input_data, output_data, label_mask))
                out_str = "|".join([str(first_layer_idx), str(day), str(start_time_slot), 
                           ";".join([":".join([",".join([str(kk) for kk in k]) for k in i_d])  for i_d in input_data]),
                           ";".join([":".join([",".join([str(kk) for kk in k]) for k in i_d])  for i_d in output_data]),
                           ";".join([":".join([str(k) for k in i_d])  for i_d in label_mask]),
                           ";".join([":".join([str(k) for k in i_d])  for i_d in label_sim_mask]),
                           ])
                writer.write(out_str+"\n")
                if max_data_size is not None and cum_num >= max_data_size:
                    break
    
    def render_data(self, first_layer_idx, raw_data):
        input_data = []
        output_data = []
        label_mask = []
        label_sim_mask = []
        for second_layer_idx, r_d in enumerate(raw_data):
            linkid = self.new_part2id_dict[(first_layer_idx, second_layer_idx)]
            x, y, mask, mask_sim = self.generate_data(linkid, r_d)
            input_data.append(x)
            output_data.append(y)
            label_mask.append(mask)
            label_sim_mask.append(mask_sim)
        return input_data, output_data, label_mask, label_sim_mask
    
    def generate_data(self, linkid, traffic_seq):
        x = []
        y = []
        mask = []
        mask_sim = []
        slot_size = int(len(traffic_seq) // self.merge_interval)
        for idx in range(slot_size):
            
            start = max(0, self.merge_interval * idx)
            end = min(self.merge_interval * (idx + 1), len(traffic_seq))
            t_s = None
            for i in range(start, end):
                if traffic_seq[i] is not None:
                    if t_s is not None:
                        t_s += traffic_seq[i]
                    else:
                        t_s = traffic_seq[i]
                    
            if t_s is None or len(t_s) < self.valid_min_size:
                x.append([0 for _ in range(self.hist_size)]) #我们只给出0
                y.append([0 for _ in range(self.hist_size)]) 
                mask.append(0)
                mask_sim.append(0)
            else:
                y_p = [0 for _ in range(self.hist_size)] #所有的桶
                for v in t_s:
                    v_idx = int(v/self.delta_speed)
                    v_idx = min(v_idx, self.hist_size-1)
                    y_p[v_idx] += 1
                y_p = [round(t * 1.0 / sum(y_p), 2) for t in y_p]
                y.append(y_p)
                mask.append(1)
                if random.random() < self.mask_rate:
                    mask_sim.append(0)
                    x.append([0 for _ in range(self.hist_size)]) #我们只给出0
                else:
                    mask_sim.append(1)
                    x.append(y_p) #我们只给出0
        return x, y, mask, mask_sim
    


class TrafficDatasetDetail():
    def _generate_(self, data_path,edge_file, graph_file, output_path,
                 mask_rate = 0.2, 
                 fill_windows = 3, 
                 time_slot_size = 12,
                 max_data_size=None):
        '''
            data_path: the file of data
            mask_rate: how many samples are set with MASK
            fill_windows: the mask strategy is to replace real value with default value (the default value is computed by 
                values in the neighboring time slot. This parameters means the maximum time slot gap. Otherwise, using limit speed)
            time_slot_size: the size of a time slot
            max_data_size: for limiting the number of data
        '''
        self.fill_windows = fill_windows
        self.mask_rate = mask_rate
        self.time_slot_size = time_slot_size
        # self.lines = []
        _, self.new2old_edge_dict, _ = pickle.load(open(edge_file,"rb"))
        _, self.second_layer_edges, new_id2part_dict, _ = pickle.load(open(graph_file,"rb"))
        self.new_part2id_dict = {(v[0],v[1]):k for k,v in new_id2part_dict.items()}
        self.speedlimit_dict = {'bridleway':10, 'construction':0, 'cycleway':15, 'footway':5, 'living_street':5,
        'motorway':50,'motorway_link':50,'path':5,'pedestrian':5,'platform':5,'primary':40,'primary_link':40,
        'raceway':30,'residential':5,'road':10,'secondary':30,'secondary_link':30,'service':10,'steps':5,
        'tertiary':10,'tertiary_link':10,'track':5,'trunk':50,'trunk_link':50,'unclassified':20,'subway':50,'rail':40}
        random.seed(os.path.getsize(data_path))
        cum_num = 0
        with open(output_path, "w") as writer:
            for line in open(data_path):
                data = line.strip().split("\t")
                first_layer_idx = int(data[0])
                day = int(data[1])
                start_time_slot = int(data[2])
                batch_data = [] 
                raw_batch_data = [data[3+t].split("|") for t in range(time_slot_size)]
                for l in range(len(raw_batch_data[0])):
                    seq = [raw_batch_data[t][l] for t in range(time_slot_size)]
                    new_seq = []
                    for s in seq:
                        if s == '':
                            new_seq.append(None)
                        else:
                            # speed, traj_num = s.split(",")
                            new_seq.append([float(speed) for speed in s.split(",")])
                    batch_data.append(new_seq)
                input_data, output_data, label_mask, label_sim_mask = self.render_data(first_layer_idx, batch_data)
                # self.lines.append((first_layer_idx, day, start_time_slot, input_data, output_data, label_mask))
                out_str = "|".join([str(first_layer_idx), str(day), str(start_time_slot), 
                           ";".join([":".join([",".join([str(kk) for kk in k]) for k in i_d])  for i_d in input_data]),
                           ";".join([":".join([",".join([str(kk) for kk in k]) for k in i_d])  for i_d in output_data]),
                           ";".join([":".join([str(k) for k in i_d])  for i_d in label_mask]),
                           ";".join([":".join([str(k) for k in i_d])  for i_d in label_sim_mask]),
                           ])
                writer.write(out_str+"\n")
                if max_data_size is not None and cum_num >= max_data_size:
                    break
    
    def render_data(self, first_layer_idx, raw_data):
        input_data = []
        output_data = []
        label_mask = []
        label_sim_mask = []
        for second_layer_idx, r_d in enumerate(raw_data):
            assert len(r_d) == self.time_slot_size
            linkid = self.new_part2id_dict[(first_layer_idx, second_layer_idx)]
            x, y, mask, mask_sim = self.generate_data(linkid, r_d)
            input_data.append(x)
            output_data.append(y)
            label_mask.append(mask)
            label_sim_mask.append(mask_sim)
        return input_data, output_data, label_mask, label_sim_mask
    
    def generate_data(self, linkid, traffic_seq):
        x = []
        y = []
        mask = []
        mask_sim = []
        for idx, t_s in enumerate(traffic_seq):
            # speed, traj_num = t_s
            if t_s is None:
                for j in range(idx, max(0,idx-self.fill_windows)-1, -1):
                    tmp_ts = traffic_seq[j]
                    if tmp_ts != None:
                        t_s = tmp_ts
                        break
                if t_s is None:
                    for j in range(idx, min(self.time_slot_size, idx+self.fill_windows+1), 1):
                        tmp_ts = traffic_seq[j]
                        if tmp_ts != None:
                            t_s = tmp_ts
                            break
                if t_s is None:
                    _, _, _, ty = self.new2old_edge_dict[linkid]
                    t_s = [round(self.speedlimit_dict.get(ty, 50) * random.random(),2)] #限速的一半
  
                # x.append(t_s)
                y.append([0, 0, 0, 0, 0]) #我们只给出0
                mask.append(0)
                mask_sim.append(0)
            else:
                y_p = [
                        round(np.min(t_s),2),
                        round(np.max(t_s),2),
                        round(np.median(t_s),2),
                        round(np.mean(t_s),2),
                        len(t_s)
                    ]
                y.append(y_p)
                mask.append(1)
                if random.random() < self.mask_rate:
                    # mask.append(1)
                    flag = True
                    for j in range(idx-1, max(0,idx-self.fill_windows)-1, -1):
                        tmp_ts = traffic_seq[j]
                        if tmp_ts != None:
                            t_s =tmp_ts
                            flag = False
                            break
                    if flag:
                        for j in range(idx+1, min(self.time_slot_size, idx+self.fill_windows+1), 1):
                            tmp_ts = traffic_seq[j]
                            if tmp_ts != None:
                                t_s = tmp_ts
                                flag = False
                                break
                    if flag:
                        _, _, _, ty = self.new2old_edge_dict[linkid]
                        speed = round(self.speedlimit_dict.get(ty, 50) * random.random(), 2) #限速的一半
                        t_s = [speed, speed, speed, speed, 0]
                    mask_sim.append(0)
                else:
                    # mask.append(0)
                    # x.append(t_s)
                    mask_sim.append(1)
            x_p = [
                        round(np.min(t_s),2),
                        round(np.max(t_s),2),
                        round(np.median(t_s),2),
                        round(np.mean(t_s),2),
                        len(t_s)
                    ]
            x.append(x_p)
        return x, y, mask, mask_sim
    

class TrafficDataset():
    def _generate_(self, data_path,edge_file, graph_file, output_path,
                 mask_rate = 0.2, 
                 fill_windows = 3, 
                 time_slot_size = 12,
                 max_data_size=None):
        '''
            data_path: the file of data
            mask_rate: how many samples are set with MASK
            fill_windows: the mask strategy is to replace real value with default value (the default value is computed by 
                values in the neighboring time slot. This parameters means the maximum time slot gap. Otherwise, using limit speed)
            time_slot_size: the size of a time slot
            max_data_size: for limiting the number of data
        '''
        self.fill_windows = fill_windows
        self.mask_rate = mask_rate
        self.time_slot_size = time_slot_size
        # self.lines = []
        _, self.new2old_edge_dict, _ = pickle.load(open(edge_file,"rb"))
        _, self.second_layer_edges, new_id2part_dict, _ = pickle.load(open(graph_file,"rb"))
        self.new_part2id_dict = {(v[0],v[1]):k for k,v in new_id2part_dict.items()}
        self.speedlimit_dict = {'bridleway':10, 'construction':0, 'cycleway':15, 'footway':5, 'living_street':5,
        'motorway':50,'motorway_link':50,'path':5,'pedestrian':5,'platform':5,'primary':40,'primary_link':40,
        'raceway':30,'residential':5,'road':10,'secondary':30,'secondary_link':30,'service':10,'steps':5,
        'tertiary':10,'tertiary_link':10,'track':5,'trunk':50,'trunk_link':50,'unclassified':20,'subway':50,'rail':40}
        random.seed(os.path.getsize(data_path))
        cum_num = 0
        with open(output_path, "w") as writer:
            for line in open(data_path):
                data = line.strip().split("\t")
                first_layer_idx = int(data[0])
                day = int(data[1])
                start_time_slot = int(data[2])
                batch_data = [] 
                raw_batch_data = [data[3+t].split("|") for t in range(time_slot_size)]
                for l in range(len(raw_batch_data[0])):
                    seq = [raw_batch_data[t][l] for t in range(time_slot_size)]
                    new_seq = []
                    for s in seq:
                        if s == '':
                            new_seq.append((None,None))
                        else:
                            speed, traj_num = s.split(",")
                            new_seq.append((float(speed),int(traj_num)))
                    batch_data.append(new_seq)
                input_data, output_data, label_mask, label_sim_mask = self.render_data(first_layer_idx, batch_data)
                # self.lines.append((first_layer_idx, day, start_time_slot, input_data, output_data, label_mask))
                out_str = "|".join([str(first_layer_idx), str(day), str(start_time_slot), 
                           ";".join([":".join([",".join([str(kk) for kk in k]) for k in i_d])  for i_d in input_data]),
                           ";".join([":".join([",".join([str(kk) for kk in k]) for k in i_d])  for i_d in output_data]),
                           ";".join([":".join([str(k) for k in i_d])  for i_d in label_mask]),
                           ";".join([":".join([str(k) for k in i_d])  for i_d in label_sim_mask]),
                           ])
                writer.write(out_str+"\n")
                if max_data_size is not None and cum_num >= max_data_size:
                    break
    
    def render_data(self, first_layer_idx, raw_data):
        input_data = []
        output_data = []
        label_mask = []
        label_sim_mask = []
        for second_layer_idx, r_d in enumerate(raw_data):
            assert len(r_d) == self.time_slot_size
            linkid = self.new_part2id_dict[(first_layer_idx, second_layer_idx)]
            x, y, mask, mask_sim = self.generate_data(linkid, r_d)
            input_data.append(x)
            output_data.append(y)
            label_mask.append(mask)
            label_sim_mask.append(mask_sim)
        return input_data, output_data, label_mask, label_sim_mask
    
    def generate_data(self, linkid, traffic_seq):
        x = []
        y = []
        mask = []
        mask_sim = []
        for idx, t_s in enumerate(traffic_seq):
            speed, traj_num = t_s
            if speed is None or traj_num is None:
                for j in range(idx, max(0,idx-self.fill_windows)-1, -1):
                    tmp_speed, tmp_traj_num = traffic_seq[j]
                    if tmp_speed != None and tmp_traj_num != None:
                        speed, traj_num = tmp_speed, tmp_traj_num
                        break
                if speed is None:
                    for j in range(idx, min(self.time_slot_size, idx+self.fill_windows+1), 1):
                        tmp_speed, tmp_traj_num = traffic_seq[j]
                        if tmp_speed != None and tmp_traj_num != None:
                            speed, traj_num = tmp_speed, tmp_traj_num
                            break
                if speed is None:
                    _, _, _, ty = self.new2old_edge_dict[linkid]
                    speed = round(self.speedlimit_dict.get(ty, 50) * random.random(),2) #限速的一半
                    traj_num = 0
                x.append([speed, traj_num])
                y.append([0, 0])
                mask.append(0)
                mask_sim.append(0)
            else:
                y.append([speed,traj_num])
                mask.append(1)
                if random.random() < self.mask_rate:
                    # mask.append(1)
                    flag = True
                    for j in range(idx, max(0,idx-self.fill_windows)-1, -1):
                        tmp_speed, tmp_traj_num = traffic_seq[j]
                        if tmp_speed != None and tmp_traj_num != None:
                            speed, traj_num = tmp_speed, tmp_traj_num
                            flag = False
                            break
                    if flag:
                        for j in range(idx, min(self.time_slot_size, idx+self.fill_windows+1), 1):
                            tmp_speed, tmp_traj_num = traffic_seq[j]
                            if tmp_speed != None and tmp_traj_num != None:
                                speed, traj_num = tmp_speed, tmp_traj_num
                                flag = False
                                break
                    if flag:
                        _, _, _, ty = self.new2old_edge_dict[linkid]
                        speed = round(self.speedlimit_dict.get(ty, 50) * random.random(), 2) #限速的一半
                        traj_num = 0 
                    x.append([speed, traj_num])
                    mask_sim.append(0)
                else:
                    # mask.append(0)
                    x.append([speed, traj_num])
                    mask_sim.append(1)
        return x, y, mask, mask_sim
    


class MyTrafficDataset(IterableDataset):
    in_dim = 4
    time_slot_num = 12
    spatial_feature_dim = 2                                              
    def __init__(self,file_path: str, edge_file:str, graph_file:str, city:str = None, unlimit:bool = False):                                                                          
        super(MyTrafficDataset).__init__()                                        
        self.file_path = file_path                                              
        self.time_slot_size, self.end  = self._get_file_info(file_path)                             
        self.start = 0
        _, self.new2old_edge_dict, _ = pickle.load(open(edge_file,"rb"))
        self.first_layer_edges, self.second_layer_edges, new_id2part_dict, _ = pickle.load(open(graph_file,"rb"))
        self.new_part2id_dict = {(v[0],v[1]):k for k,v in new_id2part_dict.items()}
        self.city = city
        self.unlimit = unlimit
                                                                                  
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:
        #     iter_start = self.start
        #     iter_end   = self.end
        # else:  # multiple workers
        #     per_worker = int(np.ceil((self.end - self.start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     iter_start = self.start + worker_id * per_worker
        #     iter_end = min(iter_start + per_worker, self.end)
        iter_start = self.start
        iter_end   = self.end
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)

        sample_iterator = self._sample_generator(iter_start, iter_end, worker_id, num_workers)
        return sample_iterator
                                                                            
    def __len__(self):
        return self.end - self.start                                            
    
    def _get_file_info(self, file_path):
        end = 0
        time_slot_size = None                                                                 
        with open(file_path, 'r') as fin:                                       
            for _, line in enumerate(fin):
                if  time_slot_size == None:
                    _, _, _, _, _, label_mask, _ = line.strip().split('|')
                    time_slot_size = len(label_mask.split(";")[0].split(":"))
                end += 1                                                                                            
        return time_slot_size, end      

    def _resort_data_and_rewrite(self, loss, data_per_batch, new_file_path):
        sorted_indices = np.argsort(loss)
        assert len(sorted_indices) == len(loss)
        data = []
        # cum_number = 0
        data_item = []
        print(len(loss), data_per_batch)
        print(len(sorted_indices))
        print(self.file_path)
        with open(self.file_path, 'r') as fin:
            for i, line in enumerate(fin):
                # cum_number += 1
                data_item.append(line.strip())
                if len(data_item) == data_per_batch:
                    data.append(data_item)
                    # cum_number = 0
                    data_item = []
                    if len(data) == len(loss):
                        break
                    
        if len(data_item) > 0:
            data.append(data_item)

        print(len(data))       
        assert len(sorted_indices) == len(data)
        with open(new_file_path, "w") as fout:
            for idx in sorted_indices:
                try:
                    data_item = data[idx]
                except Exception as err:
                    print(idx)
                    data_item = data[idx]
                for di in data_item:
                    fout.write(di+"\n")



                                                                                
    def _sample_generator(self, start, end, worker_id, num_workers):
        file_list = itertools.cycle([self.file_path]) if self.unlimit else [self.file_path]
        for file in file_list:
            with open(file, 'r') as fin:
                for i, line in enumerate(fin):    #每次读这么多量
                    if i < start:
                        continue                                          
                    if i >= end:
                        return StopIteration()
                    if i % num_workers != worker_id:
                        continue
                    items = line.strip().split('|')
                    first_layer_idx, day, start_time_slot, input_data, output_data, label_mask, sim_mask = items
                    first_layer_idx = int(first_layer_idx)
                    day = int(day)
                    start_time_slot = int(start_time_slot)
                    input_data = [[[float(kk) for kk in k.split(",")[:MyTrafficDataset.in_dim]] for k in i_d.split(":")] for i_d in input_data.split(";")]
                    output_data = [[[float(kk) for kk in k.split(",")[:MyTrafficDataset.in_dim]] for k in i_d.split(":")] for i_d in output_data.split(";")]
                    label_mask = [[int(k) for k in i_d.split(":")] for i_d in label_mask.split(";")]
                    sim_mask = [[int(k) for k in i_d.split(":")] for i_d in sim_mask.split(";")]
                    assert len(input_data) == len(output_data)
                    assert len(output_data) == len(label_mask)
                    if self.city == 'newyork':
                        weeks = [(start_time_slot+i)//24 for i in range(self.time_slot_size)]
                        minutes = [(start_time_slot+i)%24 for i in range(self.time_slot_size)]
                    else:
                        date = datetime.datetime.strptime(str(day), "%Y%m%d")
                        nxt_date = date + datetime.timedelta(days=1)
                        weeks  = [date.weekday() if (start_time_slot+i)//288 == 0 else nxt_date.weekday()  for i in range(self.time_slot_size) ]
                        minutes = [(start_time_slot+i) % 288  for i in range(self.time_slot_size)]
                    edges = self.second_layer_edges[first_layer_idx]
                    num_nodes = len(edges)
                    u = []
                    v = []
                    for node_i, edge in enumerate(edges):
                        for e in edge:
                            u.append(node_i)
                            v.append(e)
                    g = dgl.graph((u,v),num_nodes=num_nodes)
                    node_idx = [self.new_part2id_dict[(first_layer_idx, s)] for s in range(num_nodes)]
                    node_features = [self.new2old_edge_dict[i][1:3] for i in node_idx]
                    g.ndata['spatial_feature'] = torch.FloatTensor(node_features)
                    # g.ndata['traffic'] = torch.FloatTensor(input_data)
                    g.ndata['spatial_idx'] = torch.LongTensor(node_idx)
                    yield { "first_layer_idx":first_layer_idx, 
                            "weeks": weeks,
                            "minutes": minutes,
                            "graph": g,
                            "mask": label_mask,
                            "sim_mask": sim_mask,
                            "x": input_data,
                            "y": output_data }



class MyTrafficDatasetPath(IterableDataset):
    in_dim = 4
    time_slot_num = 12
    spatial_feature_dim = 2                                               
    def __init__(self,file_path: str, edge_file:str, graph_file:str, city:str = None, ratio = None):                                                                          
        super(MyTrafficDatasetPath).__init__()                                        
        self.file_path = file_path                                            
        self.time_slot_size, self.end  = self._get_file_info(file_path, ratio=ratio)                        
        self.start = 0
        _, self.new2old_edge_dict, _ = pickle.load(open(edge_file,"rb"))
        self.first_layer_edges, self.second_layer_edges, new_id2part_dict, _ = pickle.load(open(graph_file,"rb"))
        self.new_part2id_dict = {(v[0],v[1]):k for k,v in new_id2part_dict.items()}
        self.city = city
        self.ratio = ratio
                                                                                  
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = os.listdir(self.file_path)
        if self.ratio is not None:
            ratio = min(self.ratio, 1.0)
            file_number = max(int(ratio*len(files)), 1)
            files = files[:file_number]
        files_num = len(files)
        if worker_info is not None:
            assert int(worker_info.num_workers) <= files_num
            worker_id = worker_info.id
            files = [t for i, t in enumerate(files) if i % int(worker_info.num_workers) == worker_id]
        return self._sample_generator(files)
    
                                                                            
    def __len__(self):
        return self.end - self.start                                            
    
    def _get_file_info(self, file_path, ratio = None):
        end = 0
        time_slot_size = None      
        files = os.listdir(file_path)
        if ratio is not None:
            ratio = min(ratio, 1.0)
            file_number = max(int(ratio*len(files)), 1)
            files = files[:file_number]
        for file in files: 
            file_name = os.path.join(file_path, file)
            if time_slot_size == None:
                with open(file_name, 'r') as fin:                                       
                    for _, line in enumerate(fin):
                        _, _, _, _, _, label_mask, _ = line.strip().split('|')
                        time_slot_size = len(label_mask.split(";")[0].split(":"))
                        break
            t = os.popen('wc -l {}'.format(file_name)).read()
            end += int(t.strip().split(' ')[0])
                                                                                                                
        return time_slot_size, end                                                             
                                                                                
    def _sample_generator(self, filenames):
        for file in itertools.cycle(filenames):
            filename = os.path.join(self.file_path, file)
            with open(filename, 'r') as fin:
                for i, line in enumerate(fin):
                    # if i < start:
                    #     continue                                          
                    # if i >= end:
                    #     return StopIteration()
                    items = line.strip().split('|')
                    first_layer_idx, day, start_time_slot, input_data, output_data, label_mask, sim_mask = items
                    first_layer_idx = int(first_layer_idx)
                    day = int(day)
                    start_time_slot = int(start_time_slot)
                    input_data = [[[float(kk) for kk in k.split(",")[:MyTrafficDataset.in_dim]] for k in i_d.split(":")] for i_d in input_data.split(";")]
                    output_data = [[[float(kk) for kk in k.split(",")[:MyTrafficDataset.in_dim]] for k in i_d.split(":")] for i_d in output_data.split(";")]
                    label_mask = [[int(k) for k in i_d.split(":")] for i_d in label_mask.split(";")]
                    sim_mask = [[int(k) for k in i_d.split(":")] for i_d in sim_mask.split(";")]
                    assert len(input_data) == len(output_data)
                    assert len(output_data) == len(label_mask)
                    if self.city == 'newyork':
                        weeks = [(start_time_slot+i)//24 for i in range(self.time_slot_size)]
                        minutes = [(start_time_slot+i)%24 for i in range(self.time_slot_size)]
                    else:
                        date = datetime.datetime.strptime(str(day), "%Y%m%d")
                        nxt_date = date + datetime.timedelta(days=1)
                        weeks  = [date.weekday() if (start_time_slot+i)//288 == 0 else nxt_date.weekday()  for i in range(self.time_slot_size) ]
                        minutes = [(start_time_slot+i) % 288  for i in range(self.time_slot_size)]
                    edges = self.second_layer_edges[first_layer_idx]
                    num_nodes = len(edges)
                    u = []
                    v = []
                    for node_i, edge in enumerate(edges):
                        for e in edge:
                            #根据mask判断是不是需要(如果出发点没有有效的数据，那么该点不能产出)
                            if sum(label_mask[node_i]) == 0 and node_i != e:
                                continue
                            u.append(node_i)
                            v.append(e)
                    g = dgl.graph((u,v),num_nodes=num_nodes)
                    node_idx = [self.new_part2id_dict[(first_layer_idx, s)] for s in range(num_nodes)]
                    node_features = [self.new2old_edge_dict[i][1:3] for i in node_idx]
                    g.ndata['spatial_feature'] = torch.FloatTensor(node_features)
                    # g.ndata['traffic'] = torch.FloatTensor(input_data)
                    g.ndata['spatial_idx'] = torch.LongTensor(node_idx)
                    yield { "first_layer_idx":first_layer_idx, 
                            "weeks": weeks,
                            "minutes": minutes,
                            "graph": g,
                            "mask": label_mask,
                            "sim_mask": sim_mask,
                            "x": input_data,
                            "y": output_data }




# train_dataset = SummaryDataset(args.train_dataset)                                                  
# train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers) 

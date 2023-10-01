# -*- encoding: utf-8 -*-
"""
    author: yuanhaitao
    date: 2022-03-23
"""
import torch
# torch.set_default_tensor_type(torch.DoubleTensor)
import torch.nn as nn
import dgl
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
from torch.autograd import Variable
def to_var(x, requires_grad=True):
    # if torch.cuda.is_available():
    #     x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class Encoder(nn.Module):
    def __init__(self, in_dim, 
                 out_spatial_dim, 
                 out_temporal_dim, 
                 graph_layer, 
                 rnn_layer, 
                 spatial_name_str,
                 device) -> None:
        super(Encoder, self).__init__()
        # 首先是空间图的encoding，其次是时间序列的encoding
        self.spatial_conv = []
        graph_dim = in_dim
        for _ in range(graph_layer):
            self.spatial_conv.append(dglnn.GraphConv(graph_dim, out_spatial_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True).to(device))
            graph_dim = out_spatial_dim
        self.spatial_name_str = spatial_name_str
        self.temporal_rnn=nn.LSTM(input_size=in_dim, hidden_size= out_temporal_dim, num_layers=rnn_layer, batch_first=True)
        # self.reset_parameters()
        self.gelu = nn.GELU()
        self.batchnorm1 = nn.BatchNorm1d(in_dim)
    

    def forward(self, g, h):
        # h: [batch_graph_size, time_slot_size, in_dim]
        # _, time_slot_size, in_dim = h.shape
        # print("before batchnorm, top 5 h is:{}".format(h[:5]))
        # h = self.batchnorm1(h.view(-1, in_dim)).view(-1, time_slot_size, in_dim)
        # print("after batchnorm, top 5 h is:{}".format(h[:5]))
        hs_s = []
        for i in range(h.shape[1]):
            hs = h[:,i]
            # print("for {}, before conv, top 5 h is:{}".format(i, hs[:5]))
            for conv in self.spatial_conv:
                hs = self.gelu(conv(g,hs))
            # print("for {}, after conv, top 5 h is:{}".format(i, hs[:5]))
            with g.local_scope():
                g.ndata[self.spatial_name_str] = hs
                # 使用平均读出计算图表示
                hg = dgl.mean_nodes(g, self.spatial_name_str)
                # print("for {}, top 5 hg is: {}".format(i, hg[:5]))
                hs_s.append(hg)
        hs = torch.stack(hs_s, dim=1)

        with torch.backends.cudnn.flags(enabled=False):
            ht, _ = self.temporal_rnn(h)
        # print("after rnn, top 5 ht is: {}".format(ht[:5]))  
        ht = torch.mean(ht, dim=1)
        return hs, ht
    
class Decoder(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, 
                 spatial_context_dim, temporal_context_dim,
                 out_spatial_dim, 
                 out_temporal_dim,
                 graph_layer, 
                 rnn_layer,
                 device) -> None:
        super(Decoder, self).__init__()
        self.spatial_conv = []
        graph_dim = temporal_dim + spatial_context_dim
        for _ in range(graph_layer):
            self.spatial_conv.append(dglnn.GraphConv(graph_dim, out_spatial_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True).to(device))
            graph_dim = out_spatial_dim
        self.rnn_dim = spatial_dim + temporal_context_dim
        self.temporal_rnn=nn.LSTM(input_size=self.rnn_dim, hidden_size= out_temporal_dim, num_layers=rnn_layer, batch_first=True)
        self.gelu = nn.GELU()
        self.batchnorm2 = nn.BatchNorm1d(temporal_dim + spatial_context_dim + temporal_context_dim)
        self.batchnorm1 = nn.BatchNorm1d(self.rnn_dim)

        
        
    def forward(self, g, hs, ht, c_hs, c_ht):
        # hs: [batch_size, seq_len, spatial_dim]
        # ht: [batch_graph_size, temporal_dim]
        # c_hs: [batch_graph_size, spatial_context_dim]
        # c_ht: [batch_size, seq_len, temporal_context_dim]
        batch_size, seq_len, _ = hs.shape
        # hs = self.batchnorm1(torch.concat([hs, c_ht], dim=-1).view(batch_size * seq_len, -1)).view(-1, seq_len, self.rnn_dim)
        # ht = self.batchnorm2(torch.concat([ht, c_hs], dim=-1))
        hs = torch.concat([hs, c_ht], dim = -1)
        ht = torch.concat([ht, c_hs], dim = -1)
        
        for conv in self.spatial_conv:
            ht = self.gelu(conv(g,ht))
        with torch.backends.cudnn.flags(enabled=False):
            hs, _ = self.temporal_rnn(hs)
        return hs, ht
    
class EncoderDecoder(nn.Module):
    def __init__(self, slot_size, in_dim, hidden_size, spatial_feature_dim, out_spatial_dim, 
                 out_temporal_dim, 
                 graph_layer, 
                 rnn_layer, 
                 spatial_context_dim, 
                 temporal_context_dim,
                 region_nums,
                 link_nums,
                 region_edges,
                 spatial_name_str,
                 device,
                 use_global = True,
                 use_local = True,
                 use_fusion = True) -> None:
        super().__init__()
        self.encoder = Encoder(in_dim, out_spatial_dim, 
                 out_temporal_dim, 
                 graph_layer, 
                 rnn_layer, 
                 spatial_name_str,
                 device)
        
        self.use_global = use_global
        self.s2ctx = nn.Sequential(
            nn.Linear(out_temporal_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, spatial_context_dim)
        )
        self.t2ctx = nn.Sequential(
            nn.Linear(out_spatial_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, temporal_context_dim)
        )
        self.use_local = use_local
        self.ctx2t = nn.Sequential(
            nn.Linear(spatial_context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_temporal_dim)
        )
        self.ctx2s = nn.Sequential(
            nn.Linear(temporal_context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_spatial_dim)
        )
        self.use_fusion = use_fusion
        
        week_dim = max(temporal_context_dim//10, 2)
        self.week_embedding = nn.Embedding(7, week_dim)
        self.minute_embedding = nn.Embedding(288, temporal_context_dim)
        self.temporal_rnn=nn.LSTM(input_size=week_dim+temporal_context_dim, 
                                  hidden_size= spatial_context_dim, num_layers=rnn_layer, batch_first=True)
        
        self.region_embedding = nn.Embedding(region_nums, spatial_context_dim)
        self.global_regions = torch.LongTensor([_ for _ in range(region_nums)]).to(device, non_blocking=True)
        
        self.global_gcn = dglnn.GraphConv(spatial_context_dim, spatial_context_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True).to(device)
        self.global_g = dgl.graph(region_edges, num_nodes=region_nums).to(device)
        
        self.link_embedding = nn.Embedding(link_nums, spatial_context_dim)
        self.local_gcn = dglnn.GraphConv(spatial_context_dim+spatial_feature_dim, spatial_context_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True).to(device)
    
        self.gelu = nn.GELU()
        
        self.fusion = nn.Sequential(
            nn.Linear(out_spatial_dim + out_temporal_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_spatial_dim + out_temporal_dim)
        )
        self.batchnorm1 = nn.BatchNorm1d(out_spatial_dim+out_temporal_dim)
        
        self.decoder = Decoder(out_spatial_dim, 
                out_temporal_dim, 
                spatial_context_dim, 
                temporal_context_dim,
                out_temporal_dim, 
                 out_spatial_dim, 
                 graph_layer, 
                 rnn_layer,
                 device)
    
        self.extend_spatial = nn.Sequential(
            nn.Linear(spatial_context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, temporal_context_dim * slot_size)
        )
        self.extend_temporal = nn.Sequential(
            nn.Linear(spatial_context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, temporal_context_dim)
        )
        self.spatial_context_dim = spatial_context_dim
        self.temporal_context_dim = temporal_context_dim
        
        self.extend_beta = nn.Sequential(
            nn.Linear(out_temporal_dim + out_spatial_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, (out_temporal_dim+out_spatial_dim) * slot_size)
        )

        self.out_spatial_dim = out_spatial_dim

        #fusion
        self.spc2tmo = nn.Linear(spatial_context_dim, out_temporal_dim)
        self.spc2spo = nn.Linear(spatial_context_dim, out_spatial_dim)
        self.spo2tmo = nn.Linear(out_spatial_dim, out_temporal_dim)
        self.tmo2spo = nn.Linear(out_temporal_dim, out_spatial_dim)
        
        self.s2tattention = MultiHeadAttention(out_temporal_dim, out_spatial_dim, out_spatial_dim + out_temporal_dim , 1)
        self.t2sattention = MultiHeadAttention(out_spatial_dim, out_temporal_dim, out_temporal_dim + out_spatial_dim, 1)
        
        
    def forward(self, weeks, minutes, global_spatial_idx, 
                batch_local_g, traffic_h, 
                local_batch_idx,
                local_spatial_idx,
                local_spatial_feature):
        # weeks: [batch_size, seq]
        # minutes: [batch_size, seq]
        # global_spatial_idx: [batch_size]
        # batch_local_g: graph
        # traffic_h: [batch_graph_size, time_slot_size, in_dim]
        # local_batch_idx: [batch_graph_size]
        # local_spatial_idx: [batch_graph_size]
        # local_spatial_h: [batch_graph_size, spatial_context_dim]
        
        hs, ht = self.encoder(batch_local_g, traffic_h)
        bs, seq_len, _  = traffic_h.shape
        
        week_emb = self.week_embedding(weeks)
        minute_emb = self.minute_embedding(minutes)
        with torch.backends.cudnn.flags(enabled=False):
            local_temporal_h, _ = self.temporal_rnn(torch.concat([week_emb, minute_emb],dim=-1))
        
        global_temporal_h = torch.mean(local_temporal_h, dim=1)
        global_temporal_h = torch.index_select(global_temporal_h, dim=0, index=local_batch_idx)
        # print("top 5 global_temporal_h is:{}".format(global_temporal_h[:5]))
        
        global_emb = self.region_embedding(self.global_regions)
        # print("before gcn, top 5 global_emb is:{}".format(global_emb[:5]))
        global_emb = self.gelu(self.global_gcn(self.global_g, global_emb))
        # print("after gcn, top 5 global_emb is:{}".format(global_emb[:5]))
        
        global_spatial_h = torch.index_select(global_emb, dim=0, index=global_spatial_idx)
        
        # 利用RNN
        c_ht = self.extend_temporal(local_temporal_h) + self.extend_spatial(global_spatial_h).\
            view(-1,seq_len,self.temporal_context_dim)
        
        local_spatial_emb = self.link_embedding(local_spatial_idx)
        local_spatial_h = torch.concat([local_spatial_feature, local_spatial_emb], dim=-1)
        local_spatial_h =self.gelu(self.local_gcn(batch_local_g, local_spatial_h))
        c_hs = global_temporal_h + local_spatial_h

        if self.use_global is False:
            c_hs = torch.zeros_like(c_hs)
            c_ht = torch.zeros_like(c_ht)
        if self.use_local is False:
            hs = self.ctx2s(c_ht)
            ht = self.ctx2t(c_hs)

        # hs: [batch_size, seq_len, out_spatial_dim]
        # ht: [batch_graph_size, out_temporal_dim]
        # c_hs: [batch_graph_size, spatial_context_dim]
        # c_ht: [batch_size, seq_len, temporal_context_dim]
        if self.fusion:
            alpha, beta = self.decoder(batch_local_g, hs, ht, c_hs, c_ht)
            alpha = alpha + hs
            beta = beta + ht
        else:
            alpha = self.spo2tmo(hs) + self.spc2tmo(local_temporal_h)
            beta = self.tmo2spo(ht) + self.spc2spo(local_spatial_h)
 
        # alpha:[batch_size, seq, out_temporal_dim]
        # beta: [batch_graph_size, out_spatial_dim]
        # if self.fusion:
        # 
        
        alpha_new = torch.index_select(alpha, dim=0, index=local_batch_idx)

        beta_new = beta.view(-1, 1, beta.shape[-1])
        # print(alpha_new.shape, beta_new.shape)

        beta_new_new = self.s2tattention(beta_new, alpha_new, alpha_new) #+ beta_new
        alpha_new_new = self.t2sattention(alpha_new, beta_new, beta_new) #+ alpha_new

        # print(alpha_new_new.shape, beta_new_new.shape)

        beta_new = self.extend_beta(beta_new_new).view(-1, seq_len, beta_new_new.shape[-1])
        alpha_new = alpha_new_new

        # print(alpha_new.shape, beta_new.shape)
        alpha_beta = alpha_new + beta_new
        # # beta_new = torch.unsqueeze(beta, dim=1).repeat(1, seq_len, 1)
        # beta_new = self.extend_beta(beta).view(-1, seq_len, beta.shape[-1])
        
        # #融合时空表示
        # # alpha_beta =self.batchnorm1(torch.concat([alpha_new, beta_new], dim=-1).view(bs*seq_len, -1)).view(bs, seq_len, -1)
        # alpha_beta = torch.concat([alpha_new, beta_new], dim=-1)
        # # alpha_beta = alpha_new + beta_new
        # # print("before fusion, top 5 alpha_beta is:{}".format(alpha_beta[:5]))
        alpha_beta_2 = self.fusion(alpha_beta) #+ alpha_beta
        # # print("after fusion, top 5 alpha_beta is:{}".format(alpha_beta[:5]))
        
        # # output some intermediate results
        
        return alpha_beta,alpha_beta_2

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim_q, input_dim_kv, output_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        # 注意力参数
        self.W_q = nn.Linear(input_dim_q, output_dim)
        self.W_k = nn.Linear(input_dim_kv, output_dim)
        self.W_v = nn.Linear(input_dim_kv, output_dim)
        self.W_o = nn.Linear(output_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        q = self.W_q(q)  # 查询向量
        k = self.W_k(k)  # 键向量
        v = self.W_v(v)  # 值向量

        # 按照头数进行切分和重塑
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_heads, q.size(-1) // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, k.size(-1) // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, v.size(-1) // self.num_heads).transpose(1, 2)

        # print(q.shape, k.shape, v.shape)
        scores = torch.matmul(q, k.transpose(-2, -1))  # 计算注意力分数
        attention_weights = self.softmax(scores)  # 对分数进行softmax归一化

        output = torch.matmul(attention_weights, v)  # 加权求和得到输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, output.size(-1) * self.num_heads)  # 重塑输出
        output = self.W_o(output)  # 线性映射得到最终输出

        return output


    
class MetaSTC(nn.Module):
    def __init__(self, slot_size, in_dim, spatial_feature_dim, 
                 out_spatial_dim, 
                 out_temporal_dim, 
                 graph_layer, 
                 rnn_layer, 
                 spatial_context_dim, 
                 temporal_context_dim,
                 region_nums,
                 link_nums,
                 region_edges,
                 hidden_size,
                 avg_speed_tensor,
                 spatial_name_str='spatial_feature',
                 device='cpu',
                 use_global = True,
                 use_local = True, 
                 use_fusion = True) -> None:
        super(MetaSTC, self).__init__()
        self.spatial_name_str= spatial_name_str
        self.endecoder = EncoderDecoder(slot_size, in_dim,  hidden_size, spatial_feature_dim, out_spatial_dim, 
                 out_temporal_dim, 
                 graph_layer, 
                 rnn_layer, 
                 spatial_context_dim, 
                 temporal_context_dim,
                 region_nums,
                 link_nums,
                 region_edges,
                 spatial_name_str,
                 device,
                 use_global,
                 use_local,
                 use_fusion)
        
        # for the auxillary task
        # self.endecoder_2 = EncoderDecoder(
        #     slot_size, in_dim,  hidden_size, spatial_feature_dim, out_spatial_dim, 
        #          out_temporal_dim, 
        #          graph_layer, 
        #          rnn_layer, 
        #          spatial_context_dim, 
        #          temporal_context_dim,
        #          region_nums,
        #          link_nums,
        #          region_edges,
        #          spatial_name_str,
        #          device
        # )
        self.endecoder_2 = nn.Sequential(
            nn.BatchNorm1d(slot_size),
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_spatial_dim + out_temporal_dim)
            # nn.Sigmoid()
        )

        self.device = device
        self.inver_predict_fc = nn.Sequential(
            nn.Linear(out_spatial_dim + out_temporal_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2)
        )
        
        self.direct_predict_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_dim)
        )

        self.predict_fc = nn.Sequential(
            nn.Linear(out_spatial_dim + out_temporal_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_dim)
        )

        self.predict_fc_2 = nn.Sequential(
            nn.Linear(out_spatial_dim + out_temporal_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_dim)
        )
        
        # self.gate = nn.Sequential(
        #     nn.Linear(out_spatial_dim + out_temporal_dim, hidden_size),
        #     nn.GELU(),
        #     nn.Linear(hidden_size, in_dim * in_dim)
        # )
        self.reset_parameters()
        self.epsilon = 1e-6
        self.avg_speed_tensor = avg_speed_tensor
        #
        

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for name, param in self.endecoder.named_parameters():
            if "norm" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.xavier_normal_(param, gain=gain)
            else:
                nn.init.zeros_(param)
        for name, param in self.inver_predict_fc.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=gain)
            else:
                nn.init.zeros_(param)
        for name, param in self.predict_fc.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=gain)
            else:
                nn.init.zeros_(param)
        

    def first_round_parameters(self):
        parameters = list(self.endecoder.parameters()) + list(self.predict_fc.parameters())
        for para in parameters:
            yield para
    
    def get_first_names(self):
        names = []
        for name,param in self.endecoder.named_parameters():
            names.append((name,param))
        for name,param in self.predict_fc.named_parameters():
            names.append((name,param))
        return names

    def second_round_parameters(self):
        # parameters =  list(self.endecoder.parameters()) + list(self.predict_fc.parameters()) + list(self.inver_predict_fc.parameters())
        parameters = list(self.endecoder_2.parameters()) + list(self.predict_fc_2.parameters())
        for para in parameters:
            yield para

    def predict_parameters(self):
        parameters = list(self.predict_fc.parameters())
        for para in parameters:
            yield para

    def copy2param(self):
        import copy
        self.predict_fc = copy.deepcopy(self.predict_fc_2)

    def forward(self, x, first_round=True, direct=False, auxillary = False, meta = False, sim = False):
        if sim:
            batch_local_graph, traffic_h, weeks, minutes, global_spatial_idx, mask, sim_mask = x
            raw_traffic_h, traffic_h = traffic_h
            local_batch_idx = batch_local_graph.ndata['batch_idx'] #指的是哪一个region
            local_spatial_idx = batch_local_graph.ndata['spatial_idx'] #指的是哪一个原始linkid
            local_spatial_feature = batch_local_graph.ndata[self.spatial_name_str] #指的是link的固有属性
            ha =  self.avg_speed_tensor.index_select(0, local_spatial_idx).unsqueeze(1).repeat(1, traffic_h.shape[1], 1)
            raw_alpha_beta, alpha_beta = self.endecoder(weeks, minutes, global_spatial_idx, 
                    batch_local_graph, raw_traffic_h, 
                    local_batch_idx,
                    local_spatial_idx,
                    local_spatial_feature)
            
            # alpha_beta = raw_alpha_beta + alpha_beta
            alpha_beta_2 = self.endecoder_2(traffic_h - ha)

            predict1 = torch.abs((self.predict_fc(alpha_beta + raw_alpha_beta)) * (raw_traffic_h - ha) + raw_traffic_h)
            predict2 = torch.abs((self.predict_fc_2(alpha_beta_2 + alpha_beta)) * (raw_traffic_h - ha) + raw_traffic_h)
            
            # predict = torch.abs((self.predict_fc(alpha_beta)) * ha + raw_traffic_h)
            
            predict_sum1 = torch.sum(predict1, dim = -1, keepdim=True)
            predict1 = predict1 / torch.where(predict_sum1 > 0, predict_sum1, torch.ones_like(predict_sum1, device=predict_sum1.device))
            
            predict_sum2 = torch.sum(predict2, dim = -1, keepdim=True)
            predict2 = predict2 / torch.where(predict_sum2 > 0, predict_sum2, torch.ones_like(predict_sum2, device=predict_sum2.device))

            return raw_alpha_beta, alpha_beta_2, predict1, predict2
            
        
        if direct:
            batch_local_graph, traffic_h, weeks, minutes, global_spatial_idx, mask, sim_mask = x
            local_spatial_idx = batch_local_graph.ndata['spatial_idx'] #指的是哪一个原始linkid
            predict = torch.abs(self.direct_predict_fc(traffic_h))
            predict_sum = torch.sum(predict, dim = -1, keepdim=True)
            predict = predict / torch.where(predict_sum > 0, predict_sum, torch.ones_like(predict_sum, device=predict_sum.device))
            return None, predict
        elif first_round:
            batch_local_graph, traffic_h, weeks, minutes, global_spatial_idx, mask, sim_mask = x
            # scale-up traffic_h
            # traffic_h = torch.exp(traffic_h)
            # traffic_h = batch_local_graph.ndata['traffic']
            local_batch_idx = batch_local_graph.ndata['batch_idx'] #指的是哪一个region
            local_spatial_idx = batch_local_graph.ndata['spatial_idx'] #指的是哪一个原始linkid
            local_spatial_feature = batch_local_graph.ndata[self.spatial_name_str] #指的是link的固有属性
            if meta:
                alpha_beta = traffic_h[1]
                traffic_h = traffic_h[0]
                raw_traffic_h = traffic_h
            else:
                if auxillary:
                    raw_traffic_h, traffic_h = traffic_h
                    # auxillary_input = (traffic_h - raw_traffic_h)
                    ha =  self.avg_speed_tensor.index_select(0, local_spatial_idx).unsqueeze(1).repeat(1, traffic_h.shape[1], 1)
                    auxillary_input = traffic_h - ha
                    # auxillary_input = auxillary_input / torch.where(ha > 0, ha, torch.ones_like(ha, device=ha.device))
                else:
                    raw_traffic_h = traffic_h

                if auxillary:
                    # raw_alpha_beta, _ = self.endecoder(weeks, minutes, global_spatial_idx, 
                    # batch_local_graph, traffic_h, 
                    # local_batch_idx,
                    # local_spatial_idx,
                    # local_spatial_feature) 
                    alpha_beta = self.endecoder_2(auxillary_input)

                else:
                    raw_alpha_beta, alpha_beta = self.endecoder(weeks, minutes, global_spatial_idx, 
                    batch_local_graph, traffic_h, 
                    local_batch_idx,
                    local_spatial_idx,
                    local_spatial_feature)
                    alpha_beta += raw_alpha_beta
                 
                #if auxillary is False else self.endecoder_2(auxillary_input)
                # self.endecoder_2(weeks, minutes, global_spatial_idx, 
                #     batch_local_graph, traffic_h, 
                #     local_batch_idx,
                #     local_spatial_idx,
                #     local_spatial_feature)
                #self.endecoder_2(traffic_h)
                    

            # ha =  self.avg_speed_tensor.index_select(0, local_spatial_idx).unsqueeze(1).repeat(1, traffic_h.shape[1], 1)
            # ha =  self.avg_speed_tensor.index_select(0, local_spatial_idx).unsqueeze(1).repeat(1, traffic_h.shape[1], 1)
            ha =  self.avg_speed_tensor.index_select(0, local_spatial_idx).unsqueeze(1).repeat(1, traffic_h.shape[1], 1)
            
            # predict = torch.abs((self.predict_fc(alpha_beta) if auxillary is False else self.predict_fc_2(alpha_beta)) * (traffic_h - ha) + traffic_h)
            
            predict = torch.abs((self.predict_fc(alpha_beta)) * (raw_traffic_h - ha) + raw_traffic_h)
            
            predict_sum = torch.sum(predict, dim = -1, keepdim=True)
            
            predict = predict / torch.where(predict_sum > 0, predict_sum, torch.ones_like(predict_sum, device=predict_sum.device))
            
            return alpha_beta, predict
        
        else:
            # print(type(x), x[0].shape)
            rep = self.inver_predict_fc(x)
            # predict = self.predict_fc(rep)
            return rep



class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1 
        输出:
            loss值
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]])
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask 
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''        
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
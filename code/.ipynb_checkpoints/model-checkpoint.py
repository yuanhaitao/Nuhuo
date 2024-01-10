# -*- encoding: utf-8 -*-
"""
    author: yuanhaitao
    date: 2022-03-23
"""
import torch
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
        _, time_slot_size, in_dim = h.shape
        h = self.batchnorm1(h.view(-1, in_dim)).view(-1, time_slot_size, in_dim)
        hs_s = []
        for i in range(h.shape[1]):
            hs = h[:,i]
            for conv in self.spatial_conv:
                hs = self.gelu(conv(g,hs))
            with g.local_scope():
                g.ndata[self.spatial_name_str] = hs
                # 使用平均读出计算图表示
                hg = dgl.mean_nodes(g, self.spatial_name_str)
                hs_s.append(hg)
        hs = torch.stack(hs_s, dim=1)

        with torch.backends.cudnn.flags(enabled=False):
            ht, _ = self.temporal_rnn(h)
            
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
        graph_dim = temporal_dim + spatial_context_dim + temporal_context_dim
        for _ in range(graph_layer):
            self.spatial_conv.append(dglnn.GraphConv(graph_dim, out_spatial_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True).to(device))
            graph_dim = out_spatial_dim
        self.rnn_dim = spatial_dim + temporal_context_dim + spatial_context_dim
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
        hs = self.batchnorm1(torch.concat([hs, c_ht], dim=-1).view(batch_size * seq_len, -1)).view(-1, seq_len, self.rnn_dim)
        ht = self.batchnorm2(torch.concat([ht, c_hs], dim=-1))
        
        for conv in self.spatial_conv:
            ht = self.gelu(conv(g,ht))
        with torch.backends.cudnn.flags(enabled=False):
            hs, _ = self.temporal_rnn(hs)
        return hs, ht
    
class EncoderDecoder(nn.Module):
    def __init__(self, in_dim, hidden_size, spatial_feature_dim, out_spatial_dim, 
                 out_temporal_dim, 
                 graph_layer, 
                 rnn_layer, 
                 spatial_context_dim, 
                 temporal_context_dim,
                 region_nums,
                 link_nums,
                 region_edges,
                 spatial_name_str,
                 device) -> None:
        super().__init__()
        self.encoder = Encoder(in_dim, out_spatial_dim, 
                 out_temporal_dim, 
                 graph_layer, 
                 rnn_layer, 
                 spatial_name_str,
                 device)
        
        self.week_embedding = nn.Embedding(7, temporal_context_dim//10)
        self.minute_embedding = nn.Embedding(288, temporal_context_dim)
        self.temporal_rnn=nn.LSTM(input_size=temporal_context_dim//10+temporal_context_dim, 
                                  hidden_size= temporal_context_dim, num_layers=rnn_layer, batch_first=True)
        
        self.region_embedding = nn.Embedding(region_nums, spatial_context_dim)
        self.global_regions = torch.LongTensor([_ for _ in range(region_nums)]).to(device, non_blocking=True)
        
        self.global_gcn = dglnn.GraphConv(spatial_context_dim, spatial_context_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True).to(device)
        self.global_g = dgl.graph(region_edges, num_nodes=region_nums).to(device)
        
        self.link_embedding = nn.Embedding(link_nums, spatial_context_dim)
        self.local_gcn = dglnn.GraphConv(spatial_context_dim+spatial_feature_dim, spatial_context_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True).to(device)
    
        self.gelu = nn.GELU()
        
        self.fusion = nn.Sequential(
            nn.Linear(out_spatial_dim+out_temporal_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_spatial_dim+out_temporal_dim)
        )
        self.batchnorm1 = nn.BatchNorm1d(out_spatial_dim+out_temporal_dim)
        
        self.decoder = Decoder(out_spatial_dim, 
                out_temporal_dim, 
                spatial_context_dim, 
                temporal_context_dim,
                out_spatial_dim, 
                 out_temporal_dim, 
                 graph_layer, 
                 rnn_layer,
                 device)
        
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
        
        global_emb = self.region_embedding(self.global_regions)
        global_emb = self.gelu(self.global_gcn(self.global_g, global_emb))
        
        global_spatial_h = torch.index_select(global_emb, dim=0, index=global_spatial_idx)
        c_ht = torch.concat([local_temporal_h, torch.unsqueeze(global_spatial_h, \
                            dim=1).repeat(1,seq_len,1)], dim=-1)
        
        # global_spatial_h = torch.index_select(global_spatial_h, dim=0, index=local_batch_idx)
        
        # c_ht = torch.concat([local_temporal_h, torch.unsqueeze(global_spatial_h, \
        #                     dim=1).repeat(1,seq_len,1)], dim=-1)
        # c_ht = local_temporal_h
        
        local_spatial_emb = self.link_embedding(local_spatial_idx)
        local_spatial_h = torch.concat([local_spatial_feature, local_spatial_emb], dim=-1)
        # local_spatial_h = torch.concat([global_spatial_h, local_spatial_feature, local_spatial_emb], dim=-1)
        local_spatial_h =self.gelu(self.local_gcn(batch_local_g, local_spatial_h))
        
        c_hs = torch.concat([global_temporal_h, local_spatial_h], dim=-1)
        # c_hs = local_spatial_h #torch.concat([global_temporal_h, local_spatial_h], dim=-1)
        
        
        alpha, beta = self.decoder(batch_local_g, hs, ht, c_hs, c_ht)
        # alpha:[batch_size, seq, out_temporal_dim]
        # beta: [batch_graph_size, out_spatial_dim]
        alpha = alpha + hs
        beta = beta + ht
        
        alpha_new = torch.index_select(alpha, dim=0, index=local_batch_idx)
        beta_new = torch.unsqueeze(beta, dim=1).repeat(1, seq_len, 1)
        
        #融合时空表示
        alpha_beta =self.batchnorm1(torch.concat([alpha_new, beta_new], dim=-1).view(bs*seq_len, -1)).view(bs, seq_len, -1)
        alpha_beta = self.fusion(alpha_beta) + alpha_beta
        return alpha_beta
    
class MetaSTC(nn.Module):
    def __init__(self, in_dim, spatial_feature_dim, 
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
                 spatial_name_str='spatial_feature',
                 device='cpu') -> None:
        super(MetaSTC, self).__init__()
        self.spatial_name_str= spatial_name_str
        self.endecoder = EncoderDecoder(in_dim,  hidden_size, spatial_feature_dim, out_spatial_dim, 
                 out_temporal_dim, 
                 graph_layer, 
                 rnn_layer, 
                 spatial_context_dim, 
                 temporal_context_dim,
                 region_nums,
                 link_nums,
                 region_edges,
                 spatial_name_str,
                 device)
      
        # self.endecoder = nn.Sequential(
        #     nn.Linear(in_dim, hidden_size),
        #     nn.gelu(),
        #     nn.Linear(hidden_size, out_spatial_dim+out_temporal_dim)
        # )

        self.device = device
        self.inver_predict_fc = nn.Sequential(
            nn.Linear(out_spatial_dim+out_temporal_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2)
        )

        self.predict_fc = nn.Sequential(
            nn.Linear(out_spatial_dim+out_temporal_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_dim),
            nn.Tanh()
        )
        self.reset_parameters()
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
        parameters =  list(self.endecoder.parameters()) + list(self.predict_fc.parameters()) + list(self.inver_predict_fc.parameters())
        # parameters = list(self.endecoder.parameters())
        for para in parameters:
            yield para

    def forward(self, x, first_round=True):
        if first_round:
            batch_local_graph, traffic_h, weeks, minutes, global_spatial_idx = x
            # traffic_h = batch_local_graph.ndata['traffic']
            local_batch_idx = batch_local_graph.ndata['batch_idx'] #指的是哪一个region
            local_spatial_idx = batch_local_graph.ndata['spatial_idx'] #指的是哪一个原始linkid
            local_spatial_feature = batch_local_graph.ndata[self.spatial_name_str] #指的是link的固有属性
            alpha_beta = self.endecoder(weeks, minutes, global_spatial_idx, 
                    batch_local_graph, traffic_h, 
                    local_batch_idx,
                    local_spatial_idx,
                    local_spatial_feature)
            # alpha_beta = self.endecoder(traffic_h)
            predict = (1+self.predict_fc(alpha_beta)) * traffic_h
            
            return alpha_beta, predict
        else:
            # print(type(x), x[0].shape)
            rep = self.inver_predict_fc(x)
            # predict = self.predict_fc(rep)
            return rep


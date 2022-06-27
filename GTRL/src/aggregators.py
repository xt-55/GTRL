import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from utils import *
from propagations import *
from modules import *
import time
from itertools import groupby

# add
import math
import time
import torch.nn.functional as F
from hierarchical_graph_conv import GAT, SCConv

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.parameter import Parameter
from torch_scatter import scatter_mean

TIME_WINDOW = 24
PRED_LEN = 6

class FeatureEmb(nn.Module):
    def __init__(self):
        super(FeatureEmb, self).__init__()
        # time embedding
        # month,day,hour,minute,dayofweek
        self.time_emb = nn.ModuleList([nn.Embedding(feature_size, 4) for feature_size in [12, 31, 24, 4, 7]])
        for ele in self.time_emb:
            nn.init.xavier_uniform_(ele.weight.data, gain=math.sqrt(2.0))

    def forward(self, X, pa_onehot):
        B, N, T_in, F = X.size()  # (batch_size, N, T_in, F)
        X_time = torch.cat([emb(X[:, :, :, i + 4].long()) for i, emb in enumerate(self.time_emb)],
                           dim=-1)  # time F = 4*5 = 20
        X_cxt = X[..., 2:4]  # contextual features
        X_pa = X[..., :1].long()  # PA, 0,1,...,49
        pa_scatter = pa_onehot.clone()
        X_pa = pa_scatte
        r.scatter_(-1, X_pa, 1.0)  # discretize to one-hot , F = 50
        return X_cxt, X_pa, X_time


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers < 2:
            self.layers.append(GCNLayer(in_feats, n_classes, activation, dropout))
        else:
            self.layers.append(GCNLayer(in_feats, n_hidden, activation, dropout))
            for i in range(n_layers - 2):
                self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
            self.layers.append(GCNLayer(n_hidden, n_classes, activation, dropout)) # activation or None

    def forward(self, g, features=None): # no reverse
        if features is None:
            h = g.ndata['h']
        else:
            h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

 

# aggregator for event forecasting 
class aggregator_event(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=1, attn='', latend_num = 0,
                 mode=0, x_em=0, date_em=0, edge_h=0, gnn_h=0, gnn_layer=0, city_num=0, group_num=0,
                 pred_step=0, device=0, encoder = "lstm", w = None, w_init = "rand"):
        super().__init__()
        #old
        self.h_dim = h_dim  # feature
        self.latend_num = latend_num
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.se_aggr = GCN(100, h_dim, h_dim, 2, F.relu, dropout)
        #self.se_aggr = GCN(50, h_dim, h_dim, 2, F.relu, dropout)
        # self.se_aggr = GCN(100, int(h_dim/2), h_dim, 2, F.relu, dropout)

        #new
        self.mode = mode
        self.x_em = x_em
        self.date_em = date_em
        self.edge_h = edge_h
        self.gnn_h = gnn_h
        self.gnn_layer = gnn_layer
        self.city_num = city_num
        self.group_num = group_num
        self.pred_step = pred_step
        self.device = device
        self.w_init = w_init
        self.encoder = encoder
        self.new_x = w

        if self.encoder == 'self':
            self.encoder_layer = TransformerEncoderLayer(100, nhead=2, dim_feedforward=100)
            # self.x_embed = Lin(8, x_em)
            self.x_embed = Lin(100, x_em)
        elif self.encoder == 'lstm':
            self.input_LSTM = nn.LSTM(100, x_em, num_layers=1, batch_first=True)
        if self.w_init == 'rand':
            self.w = Parameter(torch.randn(city_num, group_num).to(device, non_blocking=True), requires_grad=True)
        elif self.w_init == 'group':
            self.w = Parameter(self.new_w, requires_grad=True)

        #self.w = Parameter(torch.randn(city_num, group_num).to(device, non_blocking=True), requires_grad=True)
        self.u_embed1 = nn.Embedding(12, date_em)  # month
        self.u_embed2 = nn.Embedding(7, date_em)  # week
        self.u_embed3 = nn.Embedding(24, date_em)  # hour
        #self.edge_inf = Seq(Lin(x_em * 2 + date_em * 3, edge_h), ReLU(inplace=True))
        self.edge_inf = Seq(Lin(x_em * 2, edge_h), ReLU(inplace=True))
        self.group_gnn = nn.ModuleList([NodeModel(x_em, edge_h, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.group_gnn.append(NodeModel(gnn_h, edge_h, gnn_h))
        self.global_gnn = nn.ModuleList([NodeModel(x_em + gnn_h, 100, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.global_gnn.append(NodeModel(gnn_h, 1, gnn_h))
        if self.mode == 'ag':
            self.decoder = DecoderModule(x_em, edge_h, gnn_h, gnn_layer, city_num, group_num, device)
            self.predMLP = Seq(Lin(gnn_h, 16), ReLU(inplace=True), Lin(16, 1), ReLU(inplace=True))
        if self.mode == 'full':
            self.decoder = DecoderModule(x_em, edge_h, gnn_h, gnn_layer, city_num, group_num, device)
            self.predMLP = Seq(Lin(gnn_h, 16), ReLU(inplace=True), Lin(16, self.pred_step), ReLU(inplace=True))

    # SCConv
        hid_dim = 32
        dropout = 0.5
        alpha = 0.2
        self.SCConv = SCConv(in_features=hid_dim+50, out_features=hid_dim, dropout=dropout,\
                                   alpha=alpha, latend_num=latend_num, gcn_hop = 1)

        if maxpool == 1:
            self.dgl_global_edge_f = dgl.max_edges
            self.dgl_global_node_f = dgl.max_nodes
        else:
            self.dgl_global_edge_f = dgl.mean_edges
            self.dgl_global_node_f = dgl.mean_nodes

        out_feat = int(h_dim // 2)
        self.re_aggr1 = CompGCN_dg(h_dim, out_feat, h_dim, out_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr2 = CompGCN_dg(out_feat, h_dim, out_feat, h_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        if attn == 'add':
            self.attn = Attention(h_dim, 'add')
        elif attn == 'dot':
            self.attn = Attention(h_dim, 'dot')
        else:
            self.attn = Attention(h_dim, 'general')


    def batchInput(self, x, edge_w, edge_index):
        sta_num = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])
        for i in range(edge_index.size(0)):
            edge_index[i, :] = torch.add(edge_index[i, :], i * sta_num)
        # print(edge_index.shape)
        edge_index = edge_index.transpose(0, 1)
        # print(edge_index.shape)
        edge_index = edge_index.reshape(2, -1)
        return x, edge_w, edge_index

    def forward(self, t_list, ent_embeds, rel_embeds, graph_dict):

        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []
        nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        t_list = t_list[nonzero_idx]  # usually no duplicates

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                # 拿length前10个 也就是当前time下的前7个time
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        # time2id mapping
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        # entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_g = dgl.batch(g_list)
        # if torch.cuda.is_available():
        #     move_dgl_to_cuda(batched_g)
        # a = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]).data.cpu()
        # print(a.is_cuda)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_g = batched_g.to(device) # torch.device('cuda:0')
        batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1])
        if torch.cuda.is_available():
            type_data = batched_g.edata['type'].cuda()
        else:
            type_data = batched_g.edata['type']
        batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)


        #self.re_aggr1(batched_g, False)
       # self.re_aggr2(batched_g, False)

        #['id', 'norm', 'h', 'h_o_r', 'h_s_r_o']
        f = batched_g.ndata["id"]

        ############################
        # x,edge_w,edge_index
        x = batched_g.ndata["h"]
        #edge_w =  batched_g.edata["e_h"]
        #edge_index = batched_g.edata["eid"]

        w_data = []
        for data in batched_g.edata["e_h"]:
            re = sum(data.tolist())/100
            w_data.append(re)
        #edge_w = torch.tensor(w_data).unsqueeze(0)
        edge_w = batched_g.edata["e_h"]

        # change
        edges_one = batched_g.edges()[0].unsqueeze(1)
        edges_two = batched_g.edges()[1].unsqueeze(1)
        edge_index_a = torch.cat((edges_one, edges_two), 1)
        edge_index = edge_index_a.transpose(0, 1).unsqueeze(0)

        # change
        # x = torch.cat((x1,x2), 1)
        x = x.unsqueeze(0).unsqueeze(2)

        # start
        x = x.reshape(-1, x.shape[2], x.shape[3])
        self.city_num = x.shape[0]
        if self.w_init == 'rand':
            self.w = Parameter(torch.randn(self.city_num, self.group_num).to(device, non_blocking=True), requires_grad=True)
        elif self.w_init == 'group':
            self.w = Parameter(self.new_x, requires_grad=True)
        if self.mode == 'ag':
            self.decoder = DecoderModule(self.x_em, self.edge_h, self.gnn_h, self.gnn_layer, self.city_num, self.group_num, device)
            self.predMLP = Seq(Lin(self.gnn_h, 16), ReLU(inplace=True), Lin(16, 1), ReLU(inplace=True))
        if self.mode == 'full':
            self.decoder = DecoderModule(self.x_em, self.edge_h, self.gnn_h, self.gnn_layer, self.city_num, self.group_num, device)
            self.predMLP = Seq(Lin(self.gnn_h, 16), ReLU(inplace=True), Lin(16, self.pred_step), ReLU(inplace=True))

        if self.encoder == 'self':
            x = self.encoder_layer(x)
            x = self.x_embed(x)
            # x = x.reshape(-1, self.city_num, TIME_WINDOW, x.shape[-1])
            x = x.reshape(-1, self.city_num, 1, x.shape[-1])
            x = torch.max(x, dim=-2).values
            # print(x.shape)
        elif self.encoder == 'lstm':
            _, (x, _) = self.input_LSTM(x)
            x = x.reshape(-1, self.city_num, x.shape[-1])

        # graph pooling
        w = F.softmax(self.w)
        w1 = w.transpose(0, 1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)
        # print(w.shape,x.shape)
        g_x = torch.bmm(w1, x)
        # print(g_x.shape)

        # group gnn
        # print(u_em.shape)
        for i in range(self.group_num):
            for j in range(self.group_num):
                if i == j: continue
                g_edge_input = torch.cat([g_x[:, i], g_x[:, j]], dim=-1)
                tmp_g_edge_w = self.edge_inf(g_edge_input)
                tmp_g_edge_w = tmp_g_edge_w.unsqueeze(dim=0)
                tmp_g_edge_index = torch.tensor([i, j]).unsqueeze(dim=0).to(self.device, non_blocking=True)
                if i == 0 and j == 1:
                    g_edge_w = tmp_g_edge_w
                    g_edge_index = tmp_g_edge_index
                else:
                    g_edge_w = torch.cat([g_edge_w, tmp_g_edge_w], dim=0)
                    g_edge_index = torch.cat([g_edge_index, tmp_g_edge_index], dim=0)
        g_edge_w = g_edge_w.transpose(0, 1)
        g_edge_index = g_edge_index.unsqueeze(dim=0)
        #g_edge_index = g_edge_index.repeat_interleave(u_em.shape[0], dim=0)
        g_edge_index = g_edge_index.transpose(1, 2)
        # print(g_x.shape,g_edge_w.shape,g_edge_index.shape)
        g_x, g_edge_w, g_edge_index = self.batchInput(g_x, g_edge_w, g_edge_index)
        # print(g_x.shape,g_edge_w.shape,g_edge_index.shape)
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)

        g_x = g_x.reshape(-1, self.group_num, g_x.shape[-1])
        # print(g_x.shape,self.w.shape)
        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(g_x.size(0), dim=0)
        new_x = torch.bmm(w2, g_x)
        # print(new_x.shape,x.shape)
        new_x = torch.cat([x, new_x], dim=-1)

        #edge_w = edge_w.unsqueeze(dim=-1)

        # print(new_x.shape,edge_w.shape,edge_index.shape)
        new_x, edge_w, edge_index = self.batchInput(new_x, edge_w, edge_index)
        # print(new_x.shape,edge_w.shape,edge_index.shape)
        for i in range(self.gnn_layer):
            ref = nn.Linear(400, 100)
            edge_w_new = ref(new_x)
            batched_g.edata["e_h"] = edge_w
            batched_g.ndata["h"] = edge_w_new
            self.re_aggr1(batched_g, False)
            self.re_aggr2(batched_g, False)

        # e_h
        e_h_data = batched_g.edata["e_h"]
        e_h_data_trans = torch.transpose(batched_g.edata["e_h"], 0, 1)
        re_size = rel_embeds.size()[0]
        e_h_linear = nn.Linear(e_h_data_trans.size()[1], re_size)
        e_h_data = e_h_linear(e_h_data_trans)
        e_h_data = torch.transpose(e_h_data, 0, 1)

        e_h_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, self.h_dim)
        if torch.cuda.is_available():
            e_h_seq_tensor = e_h_data.cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                e_h_seq_tensor[i, j, :] = e_h_data[time_to_idx[t.item()]]

        # h
        h_data_trans = torch.transpose(batched_g.ndata["h"], 0, 1)
        h_re_size = ent_embeds.size()[0]
        h_linear = nn.Linear(h_data_trans.size()[1], h_re_size)
        h_data = h_linear(h_data_trans)
        h_data = torch.transpose(h_data, 0, 1)

        h_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, self.h_dim)
        if torch.cuda.is_available():
            h_seq_tensor = e_h_data.cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                h_seq_tensor[i, j, :] = h_data[time_to_idx[t.item()]]

        return e_h_seq_tensor, len_non_zero, h_seq_tensor, e_h_data, h_data

    # def forward(self, t_list, ent_embeds, rel_embeds, word_embeds, graph_dict, word_graph_dict, ent_map, rel_map):
    #
    #     times = list(graph_dict.keys())
    #     times.sort(reverse=False)  # 0 to future
    #     time_unit = times[1] - times[0]
    #     time_list = []
    #     len_non_zero = []
    #     nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
    #     t_list = t_list[nonzero_idx]  # usually no duplicates
    #
    #     for tim in t_list:
    #         length = times.index(tim)
    #         if self.seq_len <= length:
    #             # 拿length前10个 也就是当前time下的前7个time
    #             time_list.append(torch.LongTensor(
    #                 times[length - self.seq_len:length]))
    #             len_non_zero.append(self.seq_len)
    #         else:
    #             time_list.append(torch.LongTensor(times[:length]))
    #             len_non_zero.append(length)
    #
    #     unique_t = torch.unique(torch.cat(time_list))
    #     t_idx = list(range(len(unique_t)))
    #     # time2id mapping
    #     time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
    #     # entity graph
    #     g_list = [graph_dict[tim.item()] for tim in unique_t]
    #     batched_g = dgl.batch(g_list)
    #     # if torch.cuda.is_available():
    #     #     move_dgl_to_cuda(batched_g)
    #     # a = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]).data.cpu()
    #     # print(a.is_cuda)
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     batched_g = batched_g.to(device) # torch.device('cuda:0')
    #     batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1])
    #     if torch.cuda.is_available():
    #         type_data = batched_g.edata['type'].cuda()
    #     else:
    #         type_data = batched_g.edata['type']
    #     batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)
    #
    #     #self.re_aggr1(batched_g, False)
    #    # self.re_aggr2(batched_g, False)
    #
    #     #['id', 'norm', 'h', 'h_o_r', 'h_s_r_o']
    #     f = batched_g.ndata["id"]
    #
    #     ############################
    #     # x,edge_w,edge_index
    #     x = batched_g.ndata["h"]
    #     #edge_w =  batched_g.edata["e_h"]
    #     #edge_index = batched_g.edata["eid"]
    #
    #     w_data = []
    #     for data in batched_g.edata["e_h"]:
    #         re = sum(data.tolist())/100
    #         w_data.append(re)
    #     #edge_w = torch.tensor(w_data).unsqueeze(0)
    #     edge_w = batched_g.edata["e_h"]
    #
    #     # change
    #     edges_one = batched_g.edges()[0].unsqueeze(1)
    #     edges_two = batched_g.edges()[1].unsqueeze(1)
    #     edge_index_a = torch.cat((edges_one, edges_two), 1)
    #     edge_index = edge_index_a.transpose(0, 1).unsqueeze(0)
    #
    #     # change
    #     # x = torch.cat((x1,x2), 1)
    #     x = x.unsqueeze(0).unsqueeze(2)
    #
    #     # start
    #     x = x.reshape(-1, x.shape[2], x.shape[3])
    #     self.city_num = x.shape[0]
    #     if self.w_init == 'rand':
    #         self.w = Parameter(torch.randn(self.city_num, self.group_num).to(device, non_blocking=True), requires_grad=True)
    #     elif self.w_init == 'group':
    #         self.w = Parameter(self.new_x, requires_grad=True)
    #     if self.mode == 'ag':
    #         self.decoder = DecoderModule(self.x_em, self.edge_h, self.gnn_h, self.gnn_layer, self.city_num, self.group_num, device)
    #         self.predMLP = Seq(Lin(self.gnn_h, 16), ReLU(inplace=True), Lin(16, 1), ReLU(inplace=True))
    #     if self.mode == 'full':
    #         self.decoder = DecoderModule(self.x_em, self.edge_h, self.gnn_h, self.gnn_layer, self.city_num, self.group_num, device)
    #         self.predMLP = Seq(Lin(self.gnn_h, 16), ReLU(inplace=True), Lin(16, self.pred_step), ReLU(inplace=True))
    #
    #     if self.encoder == 'self':
    #         x = self.encoder_layer(x)
    #         x = self.x_embed(x)
    #         # x = x.reshape(-1, self.city_num, TIME_WINDOW, x.shape[-1])
    #         x = x.reshape(-1, self.city_num, 1, x.shape[-1])
    #         x = torch.max(x, dim=-2).values
    #         # print(x.shape)
    #     elif self.encoder == 'lstm':
    #         _, (x, _) = self.input_LSTM(x)
    #         x = x.reshape(-1, self.city_num, x.shape[-1])
    #
    #     # graph pooling
    #     w = F.softmax(self.w)
    #     w1 = w.transpose(0, 1)
    #     w1 = w1.unsqueeze(dim=0)
    #     w1 = w1.repeat_interleave(x.size(0), dim=0)
    #     # print(w.shape,x.shape)
    #     g_x = torch.bmm(w1, x)
    #     # print(g_x.shape)
    #
    #     # group gnn
    #     # print(u_em.shape)
    #     for i in range(self.group_num):
    #         for j in range(self.group_num):
    #             if i == j: continue
    #             g_edge_input = torch.cat([g_x[:, i], g_x[:, j]], dim=-1)
    #             tmp_g_edge_w = self.edge_inf(g_edge_input)
    #             tmp_g_edge_w = tmp_g_edge_w.unsqueeze(dim=0)
    #             tmp_g_edge_index = torch.tensor([i, j]).unsqueeze(dim=0).to(self.device, non_blocking=True)
    #             if i == 0 and j == 1:
    #                 g_edge_w = tmp_g_edge_w
    #                 g_edge_index = tmp_g_edge_index
    #             else:
    #                 g_edge_w = torch.cat([g_edge_w, tmp_g_edge_w], dim=0)
    #                 g_edge_index = torch.cat([g_edge_index, tmp_g_edge_index], dim=0)
    #     g_edge_w = g_edge_w.transpose(0, 1)
    #     g_edge_index = g_edge_index.unsqueeze(dim=0)
    #     #g_edge_index = g_edge_index.repeat_interleave(u_em.shape[0], dim=0)
    #     g_edge_index = g_edge_index.transpose(1, 2)
    #     # print(g_x.shape,g_edge_w.shape,g_edge_index.shape)
    #     g_x, g_edge_w, g_edge_index = self.batchInput(g_x, g_edge_w, g_edge_index)
    #     # print(g_x.shape,g_edge_w.shape,g_edge_index.shape)
    #     for i in range(self.gnn_layer):
    #         g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)
    #
    #     g_x = g_x.reshape(-1, self.group_num, g_x.shape[-1])
    #     # print(g_x.shape,self.w.shape)
    #     w2 = w.unsqueeze(dim=0)
    #     w2 = w2.repeat_interleave(g_x.size(0), dim=0)
    #     new_x = torch.bmm(w2, g_x)
    #     # print(new_x.shape,x.shape)
    #     new_x = torch.cat([x, new_x], dim=-1)
    #
    #     #edge_w = edge_w.unsqueeze(dim=-1)
    #
    #     # print(new_x.shape,edge_w.shape,edge_index.shape)
    #     new_x, edge_w, edge_index = self.batchInput(new_x, edge_w, edge_index)
    #     # print(new_x.shape,edge_w.shape,edge_index.shape)
    #     for i in range(self.gnn_layer):
    #         new_x = self.global_gnn[i](new_x, edge_index, edge_w)
    #     # print(new_x.shape)
    #     if self.mode == 'ag':
    #         for i in range(self.pred_step):
    #             new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, edge_index, edge_w)
    #             tmp_res = self.predMLP(new_x)
    #             tmp_res = tmp_res.reshape(-1, self.city_num)
    #             tmp_res = tmp_res.unsqueeze(dim=-1)
    #             if i == 0:
    #                 res = tmp_res
    #             else:
    #                 res = torch.cat([res, tmp_res], dim=-1)
    #     if self.mode == 'full':
    #         new_x = self.decoder(new_x, self.w, g_edge_index, g_edge_w, edge_index, edge_w)
    #         res = self.predMLP(new_x)
    #         res = res.reshape(-1, self.city_num, self.pred_step)
    #
    #     # print(res.shape)
    #     batched_g.ndata["h"] = new_x
    #     global_node_info = dgl.mean_nodes(batched_g, 'h')
    #     embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 3 * self.h_dim)
    #     if torch.cuda.is_available():
    #         embed_seq_tensor = embed_seq_tensor.cuda()
    #     for i, times in enumerate(time_list):
    #         for j, t in enumerate(times):
    #             embed_seq_tensor[i, j, :] = global_node_info[time_to_idx[t.item()]]
    #     embed_seq_tensor = self.dropout(embed_seq_tensor)
    #
    #     return embed_seq_tensor, len_non_zero
        ############################
        #embed_seq_tensor
        #return h_sc, len_non_zero

# aggregator for actor forecasting 
class aggregator_actor(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=1, attn=''):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        # self.se_aggr = GCN(100, int(h_dim/2), h_dim, 2, F.relu, dropout)
        self.se_aggr = GCN(100, h_dim, h_dim, 2, F.relu, dropout)
        out_feat = int(h_dim // 2)
        self.re_aggr1 = CompGCN_dg(h_dim, out_feat, h_dim, out_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr2 = CompGCN_dg(out_feat, h_dim, out_feat, h_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        if attn == 'add':
            self.attn = Attention(h_dim, 'add')
        elif attn == 'dot':
            self.attn = Attention(h_dim, 'dot')
        else:
            self.attn = Attention(h_dim, 'general')

    def forward(self, t, r, r_hist, r_hist_t, ent_embeds, rel_embeds, word_embeds, f, word_graph_dict, ent_map, rel_map):
        reverse = False
        batched_g, batched_wg, len_non_zero, r_ids_graph, idx, num_non_zero = get_sorted_r_t_graphs(t, r, r_hist, r_hist_t, graph_dict, word_graph_dict, reverse=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if batched_g: 
            r_sort = r[idx]
            batched_g = batched_g.to(device) # torch.device('cuda:0')
            batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]) 
            if torch.cuda.is_available():
                type_data = batched_g.edata['type'].cuda()
            else:
                type_data = batched_g.edata['type']
            batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)
            # if torch.cuda.is_available():
            #     move_dgl_to_cuda(batched_g)
                 
            self.re_aggr1(batched_g, reverse)
            self.re_aggr2(batched_g, reverse)

            embeds_g_r = batched_g.edata.pop('e_h')
            embeds_g_r = embeds_g_r[torch.LongTensor(r_ids_graph)].data.cpu()
             
            if batched_wg:
                batched_wg = batched_wg.to(device) 
                batched_wg.ndata['h'] = word_embeds[batched_wg.ndata['id']].view(-1, word_embeds.shape[1])
                # if torch.cuda.is_available():
                #     move_dgl_to_cuda(batched_wg)
                batched_wg.ndata['h'] = self.se_aggr(batched_wg)
                
                word_ids_wg = batched_wg.ndata['id'].view(-1).cpu().tolist()
                id_dict = dict(zip(word_ids_wg, list(range(len(word_ids_wg)))))
                g_node_embs = batched_g.ndata.pop('h').data.cpu()
                g_node_ids = batched_g.ndata['id'].view(-1)
                max_query_ent = 0
                num_nodes = len(g_node_ids)
                # cpu operation for nodes
                c_g_node_ids = g_node_ids.data.cpu().numpy()
                c_unique_ent_id = list(set(c_g_node_ids))
                ent_gidx_dict = {} # entid: [[gidx],[word_idx]]
                for ent_id in c_unique_ent_id:
                    word_ids = ent_map[ent_id]
                    word_idx = []
                    for w in word_ids:
                        try:
                            word_idx.append(id_dict[w])
                        except:
                            continue
                    if len(word_idx)>1:
                        gidx = (c_g_node_ids==ent_id).nonzero()[0]
                        word_idx = torch.LongTensor(word_idx)
                        ent_gidx_dict[ent_id] = [gidx, word_idx]
                        max_query_ent = max(max_query_ent, len(word_idx))
                 
                # cpu operation for rel
                num_edges = len(embeds_g_r)
                max_query_rel = 0
                c_r_sort = r_sort.data.cpu().numpy()
                 
                type_gidx_dict_one = {} # typeid: [[gidx, word_idx]]
                for i in range(len(r_sort)):
                    type_id = c_r_sort[i]
                    word_ids = rel_map[type_id]
                    word_idx = []
                    for w in word_ids:
                        try:
                            word_idx.append(id_dict[w])
                        except:
                            continue
                    if len(word_idx)>1:
                        word_idx = torch.LongTensor(word_idx)
                        # print(i,r_ids_graph[i],'====')
                        type_gidx_dict_one[r_ids_graph[i]] = word_idx
                        max_query_rel = max(max_query_rel, len(word_idx))
                 
                 
                max_query = max(max_query_ent, max_query_rel,1)
                # initialize a batch
                wg_node_embs = batched_wg.ndata['h'].data.cpu()
                Q_mx_ent = g_node_embs.view(num_nodes , 1, self.h_dim)
                Q_mx_rel = embeds_g_r.view(num_edges , 1, self.h_dim)
                Q_mx = torch.cat((Q_mx_ent, Q_mx_rel), dim=0)
                H_mx = torch.zeros((num_nodes + num_edges, max_query, self.h_dim))
                 
                for ent in ent_gidx_dict:
                    [gidx, word_idx] = ent_gidx_dict[ent]
                    embeds = wg_node_embs.index_select(0, word_idx)
                    if len(gidx) > 1: 
                        for i in gidx:
                            H_mx[i,range(len(word_idx)),:] = embeds
                    else:
                        H_mx[gidx,range(len(word_idx)),:] = embeds
                 
                ii = num_nodes
                for e_id in type_gidx_dict_one: # some rel do not have corresponding words
                    word_idx = type_gidx_dict_one[e_id]
                    H_mx[ii,range(len(word_idx)),:] = wg_node_embs.index_select(0, word_idx)
                    ii += 1
                 
                if torch.cuda.is_available():
                    H_mx = H_mx.cuda()
                    Q_mx = Q_mx.cuda()
                
                output, weights = self.attn(Q_mx, H_mx) # output (batch,1,h_dim)
                 
                batched_g.ndata['h'] = output[:num_nodes].view(-1, self.h_dim)
                embeds_g_r = output[num_nodes:].view(-1, self.h_dim)
            g_list = dgl.unbatch(batched_g)
            node_emb_temporal = np.zeros((self.num_nodes, self.seq_len, self.h_dim))
            for i in range(len(g_list)):
                g = g_list[i]
                feature = g.ndata['h'].data.cpu().numpy()
                indices = g.ndata['id'].data.cpu().view(-1).numpy()
                node_emb_temporal[indices,i,:]  = feature

            node_emb_temporal = torch.FloatTensor(node_emb_temporal)
            if torch.cuda.is_available():
                node_emb_temporal = node_emb_temporal.cuda()
 
            embeds_split = torch.split(embeds_g_r, len_non_zero.tolist())
            embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 1 * self.h_dim)
            if torch.cuda.is_available():
                embed_seq_tensor = embed_seq_tensor.cuda()
            for i, embeds in enumerate(embeds_split): 
                embed_seq_tensor[i, torch.arange(0,len(embeds)), :] = embeds
            embed_seq_tensor = self.dropout(embed_seq_tensor)
        else:
            node_emb_temporal = None
            embed_seq_tensor = None
        return embed_seq_tensor, len_non_zero, idx, node_emb_temporal 

class DecoderModule(nn.Module):
	def __init__(self,x_em,edge_h,gnn_h,gnn_layer,city_num,group_num,device):
		super(DecoderModule, self).__init__()
		self.device = device
		self.city_num = city_num
		self.group_num = group_num
		self.gnn_layer = gnn_layer
		self.x_embed = Lin(gnn_h, x_em)
		self.group_gnn = nn.ModuleList([NodeModel(x_em,edge_h,gnn_h)])
		for i in range(self.gnn_layer-1):
			self.group_gnn.append(NodeModel(gnn_h,edge_h,gnn_h))
		self.global_gnn = nn.ModuleList([NodeModel(x_em+gnn_h,100,gnn_h)])
		for i in range(self.gnn_layer-1):
			self.global_gnn.append(NodeModel(gnn_h,100,gnn_h))

	def forward(self,x,trans_w,g_edge_index,g_edge_w,edge_index,edge_w):
		x = self.x_embed(x)
		x = x.reshape(-1,self.city_num,x.shape[-1])
		w = Parameter(trans_w,requires_grad=False).to(self.device,non_blocking=True)
		w1 = w.transpose(0,1)
		w1 = w1.unsqueeze(dim=0)
		w1 = w1.repeat_interleave(x.size(0), dim=0)
		g_x = torch.bmm(w1,x)
		g_x = g_x.reshape(-1,g_x.shape[-1])
		for i in range(self.gnn_layer):
			g_x = self.group_gnn[i](g_x,g_edge_index,g_edge_w)
		g_x = g_x.reshape(-1,self.group_num,g_x.shape[-1])
		w2 = w.unsqueeze(dim=0)
		w2 = w2.repeat_interleave(g_x.size(0), dim=0)
		new_x = torch.bmm(w2,g_x)
		new_x = torch.cat([x,new_x],dim=-1)
		new_x = new_x.reshape(-1,new_x.shape[-1])
		# print(new_x.shape,edge_w.shape,edge_index.shape)
		for i in range(self.gnn_layer):
			new_x = self.global_gnn[i](new_x,edge_index,edge_w)

		return new_x

class NodeModel(torch.nn.Module):

    def node2edge(self, x):
        # receivers = torch.matmul(rel_rec, x)
        # senders = torch.matmul(rel_send, x)
        # edges = torch.cat([senders, receivers], dim=2)
        return x

    def edge2node(self, node_num, x, rel_type):
        mask = rel_type.squeeze()
        x = x + x * (mask.unsqueeze(0))
        # rel = rel_rec.t() + rel_send.t()
        rel = torch.tensor(np.ones(shape=(node_num,x.size()[0])))
        incoming = torch.matmul(rel.to(torch.float32), x)
        return incoming / incoming.size(1)

    def __init__(self,node_h,edge_h,gnn_h, channel_dim=120, time_reduce_size=1):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(node_h+edge_h,gnn_h), ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h+gnn_h,gnn_h), ReLU(inplace=True))

        self.conv3 = nn.Conv1d(channel_dim * time_reduce_size * 2, channel_dim * time_reduce_size * 2, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(channel_dim * time_reduce_size * 2)

        self.conv4 = nn.Conv1d(channel_dim * time_reduce_size * 2, 1, kernel_size=1, stride=1)

        self.conv5 = nn.Conv1d(channel_dim * time_reduce_size * 2, channel_dim * time_reduce_size * 2, kernel_size=1, stride=1)


    def forward(self, x, edge_index, edge_attr):
        # g_x, g_edge_index, g_edge_w
        edge = edge_attr
        node_num = x.size()[0]
        #edge = edge.permute(0, 2, 1)
        #edge = F.relu(self.bn3(self.conv3(edge)))
        edge = F.relu(self.conv3(edge))

        # edge = edge.permute(0, 2, 1)
    
        # x = edge.permute(0, 2, 1)
        x = self.conv4(edge)
        # x = x.permute(0, 2, 1)
        rel_type = F.sigmoid(x)
        
        s_input_2 = self.edge2node(node_num, edge, rel_type)
        #s_input_2 = s_input_2.permute(0, 2, 1)
        #s_input_2 = F.relu(self.bn5(self.conv5(s_input_2)))
        #s_input_2 = F.relu(self.conv5(s_input_2))


        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        #row, col = edge_index
        #out = torch.cat([x[row], edge_attr], dim=1)
        #out = self.node_mlp_1(out)
        #out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        #out = torch.cat([x, out], dim=1)
        #return self.node_mlp_2(out)
        return s_input_2

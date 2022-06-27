import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from aggregators import *
from utils import *
from modules import *
import time
import math
import random
import itertools
import collections

import nni

class ConvTransR(torch.nn.Module):
    def __init__(self, num_relations, embedding_dim, input_dropout=0.2, hidden_dropout=0.2, feature_map_dropout=0.2, channels=50, kernel_size=3, use_bias=True):
        super(ConvTransR, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_relations*2)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets, nodes_id=None, mode="train", negative_rate=0):

        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        # if mode=="train":
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
        # else:
        #     e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        #     e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, e2_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, emb_rel.transpose(1, 0))
        return x



# event forecasting
class glean_event(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0,
                 use_gru=1, attn='', latend_num = 0, mode=0, x_em=0, date_em=0, edge_h=0, gnn_h=0,
                 gnn_layer=0, city_num=0, group_num=0, pred_step=0, device=0):
        super().__init__()
        # old
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.latend_num = latend_num
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))

        self.word_embeds = None
        self.global_emb = None  
        self.ent_map = None
        self.rel_map = None
        self.word_graph_dict = None
        self.graph_dict = None
        self.aggregator= aggregator_event(h_dim, dropout, num_ents, num_rels, seq_len, maxpool, attn, latend_num,
                                          mode, x_em, date_em, edge_h, gnn_h, gnn_layer, city_num, group_num,
                                          pred_step, device)
        if use_gru:
            self.encoder = nn.GRU(h_dim, h_dim, batch_first=True)
        else:
            self.encoder = nn.RNN(h_dim, h_dim, batch_first=True)
        self.linear_r = nn.Linear(h_dim, self.num_rels)

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = soft_cross_entropy
        self.init_weights()

        self.rdecoder = ConvTransR(num_rels, h_dim)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, true_prob_r, time_triples):
        pred, idx, _ = self._get_pred_embeds_new(t_list, time_triples)
        # pred, idx, _ = self.__get_pred_embeds(t_list)
        loss = self.criterion(pred, true_prob_r[idx])
        return loss

    def _get_pred_embeds_new(self, t_list, time_triples):
        sorted_t, idx = t_list.sort(0, descending=True)

        rel_em_new, len_non_zero, node_em_new, rel_1, node_1 = self.aggregator(sorted_t, self.ent_embeds,
                                                         self.rel_embeds,
                                                         self.graph_dict)

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(rel_em_new, len_non_zero, batch_first=True)
        node_packed_input = torch.nn.utils.rnn.pack_padded_sequence(node_em_new, len_non_zero, batch_first=True)
        _, rel_feature = self.encoder(packed_input)
        _, node_feature = self.encoder(node_packed_input)

        # feature = feature.squeeze(0)
        #
        # if torch.cuda.is_available():
        #     feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1)).cuda()), dim=0)
        # else:
        #     feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1))), dim=0)
        #
        # pred_x = self.linear_r(feature)

        self.layer_norm = True
        evolve_embs = node_em_new
        r_emb = rel_em_new

        evolve_embs = evolve_embs.reshape(evolve_embs.shape[0]*evolve_embs.shape[1], evolve_embs.shape[2])
        r_emb = r_emb.reshape(r_emb.shape[0]*r_emb.shape[1], r_emb.shape[2])

        tmp_ev = torch.transpose(evolve_embs, 0, 1)
        linear_rn = nn.Linear(in_features=evolve_embs.size()[0], out_features=self.ent_embeds.size()[0])
        line_sc = linear_rn(tmp_ev)
        evolve_embs = torch.transpose(line_sc, 1, 0)

        tmp_r = torch.transpose(r_emb, 0, 1)
        linear_rs = nn.Linear(in_features=r_emb.size()[0], out_features=self.num_rels)
        r_emb_data = linear_rs(tmp_r)
        r_emb = torch.transpose(r_emb_data, 1, 0)

        pre_emb = F.normalize(evolve_embs if self.layer_norm else evolve_embs[-1])
        score_rel = self.rdecoder.forward(pre_emb, r_emb, time_triples, mode="train").view(-1, self.num_rels)

        tmp_sc = torch.transpose(score_rel, 0, 1)
        linear_rn = nn.Linear(in_features=time_triples.size()[0], out_features=len(t_list))
        line_sc = linear_rn(tmp_sc)
        score_rel = torch.transpose(linear_rn(tmp_sc), 1, 0)

        score_rel_tpm = []
        score_rel_new = [[]]
        for data in score_rel.tolist():
            for i in range(0, len(data)):
                if len(score_rel_tpm) < i+1:
                    score_rel_tpm.append([])
                score_rel_tpm[i].append(data[i])
        for score_list in score_rel_tpm:
            score_rel_new[0].append(sum(score_list)/len(score_list))
        score_rel_new = torch.tensor(score_rel_new, dtype=torch.double)

        return score_rel, idx, None

    def __get_pred_embeds(self, t_list):
        sorted_t, idx = t_list.sort(0, descending=True)  
        embed_seq_tensor, len_non_zero = self.aggregator(sorted_t, self.ent_embeds, 
                                    self.rel_embeds, self.word_embeds, 
                                    self.graph_dict, self.word_graph_dict, 
                                    self.ent_map, self.rel_map)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True)
        _, feature = self.encoder(packed_input)
        feature = feature.squeeze(0)

        if torch.cuda.is_available():
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1)).cuda()), dim=0)
        else:
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1))), dim=0)
        
        pred = self.linear_r(feature)
        return pred, idx, feature
        
    def predict(self, t_list, true_prob_r, time_triples):
        pred, idx, feature = self._get_pred_embeds_new(t_list, time_triples)
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx])
        else:
            loss = None
        return loss, pred, feature

    def evaluate(self, t, true_prob_r, time_triples):
        loss, pred, _ = self.predict(t, true_prob_r, time_triples)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        # if torch.cuda.is_available():
        #     sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        # else:
        #     sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # for data in sorted_prob_rel:
        #     print(data)

        # target
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]

        ranks = torch.LongTensor([])
        if torch.cuda.is_available():
            ranks = ranks.cuda()

        for sub_idx in nonzero_rel_idx:
            if prob_rel[sub_idx] <= 0.99999:
                continue
            rank = (prob_rel_idx == sub_idx).nonzero(as_tuple=False).view(-1)
            ranks = torch.cat((ranks, rank))

        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss, ranks

    def evaluate_1(self, t_data, r_data, r_hist, r_hist_t, true_s, true_o):
        loss, sub_pred, ob_pred = self.predict(t_data, r_data, r_hist, r_hist_t, true_s, true_o)
        # s target
        true_prob_s = true_s.view(-1)
        nonzero_sub_idx = torch.nonzero(true_prob_s, as_tuple=False)  # (x,1)->(x)
        sort_true_sub, true_sub_idx = true_prob_s.sort(0, descending=True)
        nonzero_true_sub_idx = true_sub_idx[:len(nonzero_sub_idx)]
        # o target
        true_prob_o = true_o.view(-1)
        nonzero_ob_idx = torch.nonzero(true_prob_o, as_tuple=False)  # (x,1)->(x)
        sort_true_ob, true_ob_idx = true_prob_o.sort(0, descending=True)
        nonzero_true_ob_idx = true_ob_idx[:len(nonzero_ob_idx)]

        prob_sub = self.out_func(sub_pred.view(-1))
        sort_prob_sub, prob_sub_idx = prob_sub.sort(0, descending=True)
        if torch.cuda.is_available():
            sort_prob_sub = torch.where(sort_prob_sub > self.threshold, sort_prob_sub,
                                        torch.zeros(sort_prob_sub.size()).cuda())
        else:
            sort_prob_sub = torch.where(sort_prob_sub > self.threshold, sort_prob_sub,
                                        torch.zeros(sort_prob_sub.size()))
        nonzero_prob_idx = torch.nonzero(sort_prob_sub, as_tuple=False).view(-1)
        nonzero_prob_sub_idx = prob_sub_idx[:len(nonzero_prob_idx)]

        ranks = torch.LongTensor([])
        if torch.cuda.is_available():
            ranks = ranks.cuda()
        for sub_idx in nonzero_sub_idx:
            rank = (prob_sub_idx == sub_idx).nonzero(as_tuple=False).view(-1)
            ranks = torch.cat((ranks, rank))

        ## o
        prob_ob = self.out_func(ob_pred.view(-1))
        sort_prob_ob, prob_ob_idx = prob_ob.sort(0, descending=True)
        if torch.cuda.is_available():
            sort_prob_ob = torch.where(sort_prob_ob > self.threshold, sort_prob_ob,
                                       torch.zeros(sort_prob_ob.size()).cuda())
        else:
            sort_prob_ob = torch.where(sort_prob_ob > self.threshold, sort_prob_ob, torch.zeros(sort_prob_ob.size()))
        nonzero_prob_idx = torch.nonzero(sort_prob_ob, as_tuple=False).view(-1)
        nonzero_prob_ob_idx = prob_ob_idx[:len(nonzero_prob_idx)]

        for ob_idx in nonzero_ob_idx:
            rank = (prob_ob_idx == ob_idx).nonzero(as_tuple=False).view(-1)
            ranks = torch.cat((ranks, rank))

        return nonzero_true_sub_idx, nonzero_prob_sub_idx, nonzero_true_ob_idx, nonzero_prob_ob_idx, ranks, loss


# actor forecasting
class glean_actor(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_gru=1, attn=''):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents # number of nodes
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.maxpool = maxpool
        self.dropout = nn.Dropout(dropout)
        self.rel_embeds = nn.Parameter(torch.Tensor(1*num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))
        self.W = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.W2 = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.W_ob = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.W2_ob = nn.Parameter(torch.Tensor(h_dim, h_dim))
        if use_gru:
            self.encoder1 = nn.GRU(1*h_dim, h_dim, batch_first=True)
            self.encoder2 = nn.GRU(1*h_dim, h_dim, batch_first=True)
        else:
            self.encoder1 = nn.RNN(1*h_dim, h_dim, batch_first=True)
            self.encoder2 = nn.RNN(1*h_dim, h_dim, batch_first=True)
        self.aggregator = aggregator_actor(h_dim, dropout, num_ents, num_rels, seq_len, maxpool, attn)
        self.linear_node = nn.Linear(1 * h_dim, 1)
        self.zero_linear = nn.Linear(1,1)
        self.bn = nn.LayerNorm(h_dim)
        self.graph_dict = None
        self.word_graph_dict = None
        self.global_emb = None
        self.ent_map = None
        self.rel_map = None
        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = soft_cross_entropy
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_data, r_data, r_hist, r_hist_t, true_s, true_o, reverse=False):
        if reverse: # predict s 
            soft_target = true_o
        else: 
            soft_target = true_s
        sub_pred, idx, _ = self.__get_pred_embeds(t_data, r_data, r_hist, r_hist_t, reverse)
        loss_p_sub = self.criterion(sub_pred, soft_target[idx])
        return loss_p_sub 
        
    def predict(self, t_data, r_data, r_hist, r_hist_t, true_s, true_o):
        # predict s  
        sub_pred, idx, _ = self.__get_pred_embeds(t_data, r_data, r_hist, r_hist_t, False)
        loss_sub = self.criterion(sub_pred, true_s[idx])
        
        # predict o
        ob_pred, idx, _ = self.__get_pred_embeds(t_data, r_data, r_hist, r_hist_t, True)
        loss_ob = self.criterion(ob_pred, true_o[idx])
        
        loss = loss_sub + loss_ob
        return loss, sub_pred, ob_pred  

    def __get_pred_embeds(self, t_data, r_data, r_hist, r_hist_t, reverse):
        if reverse:
            W1 = self.W_ob
            W2 = self.W2_ob
        else:
            W1 = self.W
            W2 = self.W2
        embed_seq_tensor, len_non_zero, idx, node_emb_temporal  = self.aggregator(t_data, r_data, r_hist, r_hist_t, 
                                self.ent_embeds, self.rel_embeds, self.word_embeds, self.graph_dict, 
                                self.word_graph_dict, self.ent_map, self.rel_map)
        static_embeds = torch.zeros(len(r_data), self.num_ents) # 1-d value
        if torch.cuda.is_available():
            static_embeds = static_embeds.cuda()
        r_sorted = r_data[idx]
        for i in range(len(r_sorted)):
            static_embeds[i,:] = self.ent_embeds @ W1 @ self.rel_embeds[r_sorted[i]] # all
        
        if embed_seq_tensor is not None:
            _, feature = self.encoder1(embed_seq_tensor)
            feature = feature.squeeze(0)
            if torch.cuda.is_available():
                feature = torch.cat((feature, torch.zeros(len(r_data) - len(feature), feature.size(-1)).cuda()), dim=0)
            else:
                feature = torch.cat((feature, torch.zeros(len(r_data) - len(feature), feature.size(-1))), dim=0)
            
            _, node_feature = self.encoder2(node_emb_temporal)
            node_feature = node_feature.squeeze(0) 
            pred = feature @ W2 @ node_feature.t()
            static_embeds += pred          
            
        return static_embeds, idx, None

    def evaluate(self, t_data, r_data, r_hist, r_hist_t, true_s, true_o):
        loss, sub_pred, ob_pred = self.predict(t_data, r_data, r_hist, r_hist_t, true_s, true_o)
        # s target
        true_prob_s = true_s.view(-1)  
        nonzero_sub_idx = torch.nonzero(true_prob_s,as_tuple=False) # (x,1)->(x)
        sort_true_sub, true_sub_idx = true_prob_s.sort(0, descending=True)
        nonzero_true_sub_idx = true_sub_idx[:len(nonzero_sub_idx)]
        # o target
        true_prob_o = true_o.view(-1)  
        nonzero_ob_idx = torch.nonzero(true_prob_o,as_tuple=False) # (x,1)->(x)
        sort_true_ob, true_ob_idx = true_prob_o.sort(0, descending=True)
        nonzero_true_ob_idx = true_ob_idx[:len(nonzero_ob_idx)]

        prob_sub = self.out_func(sub_pred.view(-1))
        sort_prob_sub, prob_sub_idx = prob_sub.sort(0, descending=True)
        if torch.cuda.is_available():
            sort_prob_sub = torch.where(sort_prob_sub > self.threshold, sort_prob_sub, torch.zeros(sort_prob_sub.size()).cuda())
        else:
            sort_prob_sub = torch.where(sort_prob_sub > self.threshold, sort_prob_sub, torch.zeros(sort_prob_sub.size()))
        nonzero_prob_idx = torch.nonzero(sort_prob_sub,as_tuple=False).view(-1)
        nonzero_prob_sub_idx = prob_sub_idx[:len(nonzero_prob_idx)]

        ranks = torch.LongTensor([])
        if torch.cuda.is_available():
            ranks = ranks.cuda()
        for sub_idx in nonzero_sub_idx:
            rank = (prob_sub_idx == sub_idx).nonzero(as_tuple=False).view(-1)
            ranks = torch.cat((ranks, rank))

        ## o
        prob_ob = self.out_func(ob_pred.view(-1))
        sort_prob_ob, prob_ob_idx = prob_ob.sort(0, descending=True)
        if torch.cuda.is_available():
            sort_prob_ob = torch.where(sort_prob_ob > self.threshold, sort_prob_ob, torch.zeros(sort_prob_ob.size()).cuda())
        else:
            sort_prob_ob = torch.where(sort_prob_ob > self.threshold, sort_prob_ob, torch.zeros(sort_prob_ob.size()))
        nonzero_prob_idx = torch.nonzero(sort_prob_ob,as_tuple=False).view(-1)
        nonzero_prob_ob_idx = prob_ob_idx[:len(nonzero_prob_idx)]

        for ob_idx in nonzero_ob_idx:
            rank = (prob_ob_idx == ob_idx).nonzero(as_tuple=False).view(-1)
            ranks = torch.cat((ranks, rank))
        
        return nonzero_true_sub_idx, nonzero_prob_sub_idx, nonzero_true_ob_idx, nonzero_prob_ob_idx, ranks, loss

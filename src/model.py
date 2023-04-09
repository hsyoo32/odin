#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class ODIN(nn.Module):

    def __init__(self, num_nodes, flags_obj):
        nn.Module.__init__(self)
        embedding_size = flags_obj.embedding_size
        self.srcs_int = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.tars_int = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.srcs_auth = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.tars_auth = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.srcs_hub = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.tars_hub = Parameter(torch.FloatTensor(num_nodes, embedding_size))

        self.disen_weight = flags_obj.disen_weight
        self.edge_weight = flags_obj.edge_weight
        self.option = flags_obj.option

        self.init_params()

    def init_params(self):
        stdv = 1. / math.sqrt(self.srcs_int.size(1))
        self.srcs_int.data.uniform_(-stdv, stdv)
        self.tars_int.data.uniform_(-stdv, stdv)
        self.srcs_auth.data.uniform_(-stdv, stdv)
        self.tars_auth.data.uniform_(-stdv, stdv)
        self.srcs_hub.data.uniform_(-stdv, stdv)
        self.tars_hub.data.uniform_(-stdv, stdv)

    def bpr_loss(self, p_score, n_score):

        if 'softplus' in self.option:
            loss = torch.mean(torch.nn.functional.softplus(n_score - p_score))
        else:
            loss = -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

        return loss

    def mask_bpr_loss(self, p_score, n_score, mask):

        if 'softplus' in self.option:
            loss = torch.mean(mask*torch.nn.functional.softplus(n_score - p_score))       
        else:
            loss = -torch.mean(mask*torch.log(torch.sigmoid(p_score - n_score)))

        return loss

    def forward(self, src_p, src_n, tar_p, tar_n, mask_int_tar, mask_int_src, mask_auth_up, mask_auth_down, 
                mask_hub_up, mask_hub_down, mask_hub_tar, mask_auth_src):

        srcs_p_int = self.srcs_int[src_p]
        srcs_p_auth = self.srcs_auth[src_p]
        srcs_p_hub = self.srcs_hub[src_p]
        
        tars_p_int = self.tars_int[tar_p]
        tars_p_auth = self.tars_auth[tar_p]
        tars_p_hub = self.tars_hub[tar_p]

        srcs_n_int = self.srcs_int[src_n]
        srcs_n_auth = self.srcs_auth[src_n]
        srcs_n_hub = self.srcs_hub[src_n]

        tars_n_int = self.tars_int[tar_n]
        tars_n_auth = self.tars_auth[tar_n]
        tars_n_hub = self.tars_hub[tar_n]

        p_score_int = torch.sum(srcs_p_int*tars_p_int, 2)
        n_score_int_src = torch.sum(srcs_n_int*tars_p_int, 2)
        n_score_int_tar = torch.sum(srcs_p_int*tars_n_int, 2)

        p_score_auth = torch.sum(srcs_p_auth*tars_p_auth, 2)
        n_score_auth_src = torch.sum(srcs_n_auth*tars_p_auth, 2)
        n_score_auth_tar = torch.sum(srcs_p_auth*tars_n_auth, 2)

        p_score_hub = torch.sum(srcs_p_hub*tars_p_hub, 2)
        n_score_hub_src = torch.sum(srcs_n_hub*tars_p_hub, 2)
        n_score_hub_tar = torch.sum(srcs_p_hub*tars_n_hub, 2)

        p_score_edge_src = p_score_int + p_score_auth + p_score_hub
        n_score_edge_src = n_score_int_src + n_score_auth_src + n_score_hub_src
        p_score_edge_tar = p_score_edge_src
        n_score_edge_tar = n_score_int_tar + n_score_auth_tar + n_score_hub_tar
        
        loss_edge_tar = self.bpr_loss(p_score_edge_tar, n_score_edge_tar)
        loss_auth = self.mask_bpr_loss(p_score_auth, n_score_auth_tar, mask_auth_up) \
            + self.mask_bpr_loss(n_score_auth_tar, p_score_auth, mask_auth_down)
        loss_int_hub = self.mask_bpr_loss(p_score_int, n_score_int_tar, mask_int_tar) \
            + self.mask_bpr_loss(p_score_hub, n_score_hub_tar, mask_hub_tar)
        
        loss_edge_src = self.bpr_loss(p_score_edge_src, n_score_edge_src)
        loss_hub = self.mask_bpr_loss(p_score_hub, n_score_hub_src, mask_hub_up) \
            + self.mask_bpr_loss(n_score_hub_src, p_score_hub, mask_hub_down)
        loss_int_auth = self.mask_bpr_loss(p_score_int, n_score_int_src, mask_int_src) \
            + self.mask_bpr_loss(p_score_auth, n_score_auth_src, mask_auth_src)
        
        if 'edge-auth' in self.option:
            loss = loss_edge_tar
        elif 'edge-hub' in self.option:
            loss = loss_edge_src
        elif 'component-hub' in self.option:
            loss = loss_edge_src + self.disen_weight * (loss_hub + loss_int_auth)
        elif 'component-auth' in self.option:
            loss = loss_edge_tar + self.disen_weight * (loss_auth + loss_int_hub)
        # ODIN
        else:
            loss = loss_edge_src + loss_edge_tar + self.disen_weight * (loss_hub + loss_int_auth + loss_auth + loss_int_hub)

        return loss

    def get_item_embeddings(self):

        item_embeddings = torch.cat((self.tars_int, self.tars_auth, self.tars_hub), 1)
        return item_embeddings.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        user_embeddings = torch.cat((self.srcs_int, self.srcs_auth, self.srcs_hub), 1)
        return user_embeddings.detach().cpu().numpy().astype('float32')

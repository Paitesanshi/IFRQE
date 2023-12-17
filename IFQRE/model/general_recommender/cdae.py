# -*- coding: utf-8 -*-
# @Time   : 2020/12/12
# @Author : Xingyu Pan
# @Email  : panxy@ruc.edu.cn

r"""
CDAE
################################################
Reference:
    Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. In WSDM 2016.
   
Reference code:
    https://github.com/jasonyaw/CDAE
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
class GameDistribution(nn.Module):
    def __init__(self, n_users, n_items, action_len, history, history_len, unwill, config):
        super(GameDistribution, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.action_len = action_len
        self.history = history
        self.history_len = history_len
        self.unwill = unwill
        self.distribution_type = config['distribution_type']
        self.action_type = config['action_type']
        self.gamma = torch.rand(self.n_users, dtype=torch.float32, requires_grad=True)
        self.max_action = torch.max(self.action_len)
        self.distribution = torch.zeros(self.n_users, self.max_action)
        self.action = torch.BoolTensor(self.n_users, self.n_items)
        self.action_num = list(range(self.n_users))
        self.softmax = torch.nn.Softmax(dim=0)
        self.method = config['method_type']
        self.s=0.9
        self.init_distribution()

    def init_distribution(self):
        if self.method == 'greedy':
            if self.distribution_type == 1:
                for i in range(self.n_users):
                    b = torch.zeros(self.action_len[i])
                    for j in range(self.action_len[i]):
                        for k in range(self.history_len[i]):
                            if (j >> k) & 1 == 1:
                                b[j] = b[j] + self.unwill[i, self.history[i][k]]

                        b[j] = 1 / b[j] if b[j] != 0 else 0
                    self.distribution[i][:self.action_len[i]]=self.softmax(b)
            elif self.distribution_type == 2:
                for i in range(self.n_users):
                    b = torch.zeros(self.action_len[i])
                    for j in range(self.action_len[i]):
                        for k in range(self.history_len[i]):
                            if (j >> k) & 1 == 1:
                                b[j] = b[j] + self.unwill[i, self.history[i][k]]
                        b[j] = 1 / b[j] if b[j] != 0 else 0
                    self.distribution[i][:self.action_len[i]]=self.softmax(b)
            else:
                for i in range(self.n_users):
                    self.distribution[i]=1 / self.action_len[i]
        else:
            if self.distribution_type == 1:
                for i in range(self.n_users):
                    b = torch.rand(self.action_len[i])
                    self.distribution[i][:self.action_len[i]]=self.softmax(b)
            elif self.distribution_type == 2:
                for i in range(self.n_users):
                    for j in range(self.action_len[i]):
                        cnt=bin(j).count('1')
                        self.distribution[i][j]=pow(self.s,cnt)*pow((1-self.s),(self.history_len[i]-cnt))

            else:
                for i in range(self.n_users):
                    self.distribution.fill_(1 / self.action_len[i])

    def forward(self, u):
        gam = self.sigmoid(self.gamma[u])
        temp = torch.zeros(max(self.action_len))
        if self.distribution_type == 1:
            for j in range(self.action_len[u]):
                l = bin(j).count('1')
                # temp[j] = torch.pow(self.gamma[u], l) * torch.pow(1 - self.gamma[u], self.history_len[u] - l)
                temp[j] = torch.pow(gam, l) * torch.pow(1 - gam, self.history_len[u] - l)
        elif self.distribution_type == 2:
            temp = self.gamma[u] / self.action_len[u]
        else:
            for j in range(self.action_len[u]):
                l = bin(j).count('1')
                temp[j] = torch.pow(self.gamma[0], l) * torch.pow(1 - self.gamma[0], self.history_len[u] - l)
        return temp

    def softnorm(self):
        self.distribution = self.softmax(self.distribution)
        return self.distribution

    def get_action_by_distribution(self):
        self.action.fill_(0)
        id=list()
        if self.action_type == 1:
            for i in range(self.n_users):
                o = random.randint(0, self.action_len[i] - 1)
                id.append(o)
                for j in range(self.history_len[i]):
                    self.action[i, self.history[i][j].item()] = (o.item() >> j) & 1
        elif self.action_type == 2:
            for i in range(self.n_users):
                o = torch.argmax(self.distribution[i])
                #self.action_num.append(o)
                id.append(o)
                for j in range(self.history_len[i]):
                    if (o.item() >> j) & 1:
                        self.action[i, self.history[i][j].item()] = 1

        elif self.action_type == 3:
            o = torch.zeros(self.n_users, self.n_items)
            for i in range(self.n_users):
                for k in range(self.action_len[i]):
                    for j in range(self.history_len[i]):
                        if (k >> j) & 1 == 1:
                            o[i, self.history[i][j].item()] = o[i, self.history[i][j].item()] + self.distribution[i][k]
                self.action = o > 0.5
                mi = 1
                act = 0
                for j in range(self.history_len[i]):
                    if self.action[i, j] != 0:
                        act += mi
                    mi *= 2
                id.append(act)
                #self.action_num[i] = act
        return self.action,id

    def calculate_loss(self, u, lr, del_a, d):
        loss = torch.sum(del_a * d) + (1 / lr) * torch.sum(pow(del_a - d, 2))
        return loss

class CDAE(GeneralRecommender):
    r"""Collaborative Denoising Auto-Encoder (CDAE) is a recommendation model 
    for top-N recommendation that utilizes the idea of Denoising Auto-Encoders.
    We implement the the CDAE model with only user dataloader.
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(GameCDAE, self).__init__(config, dataset)

        self.reg_weight_1 = config['reg_weight_1']
        self.reg_weight_2 = config['reg_weight_2']
        self.loss_type = config['loss_type']
        self.hid_activation = config['hid_activation']
        self.out_activation = config['out_activation']
        self.embedding_size = config['embedding_size']
        self.corruption_ratio = config['corruption_ratio']

        self.LABEL = config['LABEL_FIELD']
        self.distribution_type = config['distribution_type']
        self.action_type = config['action_type']
        self.unwillingness_type = config['unwillingness_type']
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

        if self.hid_activation == 'sigmoid':
            self.h_act = nn.Sigmoid()
        elif self.hid_activation == 'relu':
            self.h_act = nn.ReLU()
        elif self.hid_activation == 'tanh':
            self.h_act = nn.Tanh()
        else:
            raise ValueError('Invalid hidden layer activation function')

        if self.out_activation == 'sigmoid':
            self.o_act = nn.Sigmoid()
        elif self.out_activation == 'relu':
            self.o_act = nn.ReLU()
        else:
            raise ValueError('Invalid output layer activation function')

        self.dropout = nn.Dropout(p=self.corruption_ratio)

        self.h_user = nn.Embedding(self.n_users, self.embedding_size)
        self.h_item = nn.Linear(self.n_items, self.embedding_size)
        self.out_layer = nn.Linear(self.embedding_size, self.n_items)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.history, self.history_value, self.history_len = dataset.history_item_matrix()
        self.max_item = torch.max(self.history_len).item()
        self.max_action = pow(2, self.max_item)
        self.strategy_mask = torch.zeros(self.n_users, self.n_items)
        self.pow_mask = torch.zeros(self.n_users, self.n_items)
        self.action_len = pow(2, self.history_len)
        self.unwillingness = torch.zeros(self.n_users, self.n_items)
        #self.init_unwillingness()
        self.distribution = GameDistribution(self.n_users, self.n_items, self.action_len, self.history,
                                             self.history_len,
                                             self.unwillingness, config)
        self.p = torch.ones(20)
        for i in range(1, 20):
            self.p[i] = self.p[i - 1] * 2
        for i in range(self.n_users):
            for j in range(self.history_len[i]):

                self.strategy_mask[i, self.history[i][j].item()] = 1
                self.pow_mask[i, self.history[i][j].item()] = pow(2, j)

    def get_random_action(self,threshold):
        o = torch.rand(self.n_users, self.n_items)
        o = o * self.strategy_mask
        o = o >threshold
        id = torch.sum(o * self.pow_mask, dim=1).int()
        return o, id

    def get_sample_action(self, L):
        ol = list()
        ol_id = list()
        p = torch.ones(20)

        for l in range(L):
            o, id = self.get_random_action()
            ol.append(o)
            ol_id.append(id)
        return ol, ol_id

    def init_unwillingness(self,dataset):
        if self.unwillingness_type == 1:
            for i in range(self.n_users):
                self.unwillingness[i] = torch.ones(self.n_items)/2


        elif self.unwillingness_type == 2:
            for i in range(self.n_users):
                if self.history_len[i] > 0:
                    self.unwillingness[i] = torch.rand(self.n_items)

        elif self.unwillingness_type == 3:
            for i in range(self.n_users):
                for k in range(self.history_len[i]):
                    self.unwillingness[i,self.history[i][k].item()] = 1/self.history_value[k]
        elif self.unwillingness_type == 4:
            for i in range(1,self.n_users):
                categories=dict()
                minv=50
                maxv=0
                for k in range(self.history_len[i]):
                    cates=dataset.item_feat.interaction['categories'][self.history[i][k].item()]
                    cates=torch.unique(cates)
                    for j in range(len(cates)):
                        if cates[j].item()==0:
                            continue
                        if cates[j].item() in categories.keys():
                            categories[cates[j].item()]+=1
                        else:
                            categories[cates[j].item()]= 1
                        if categories[cates[j].item()]>maxv:
                            maxv=categories[cates[j].item()]
                        if categories[cates[j].item()]<minv:
                            minv=categories[cates[j].item()]
                if minv==maxv:
                    for k in range(self.history_len[i]):
                        self.unwillingness[i, self.history[i][k].item()]=0.5
                else:
                    for k in range(self.history_len[i]):
                        maxk=0
                        cates = dataset.item_feat.interaction['categories'][self.history[i][k].item()]
                        cates = torch.unique(cates)
                        for j in range(len(cates)):
                            if categories[cates[j].item()]>maxk:
                                maxk=categories[cates[j].item()]
                        self.unwillingness[i,self.history[i][k].item()] =(1/maxk-1/maxv)/ (1/minv-1/maxv)/2+0.5
                        #print(self.unwillingness[i,self.history[i][k].item()],maxk,maxv,minv)
        return self.unwillingness * self.strategy_mask

    def forward(self, x_items, x_users):
        h_i = self.dropout(x_items)
        h_i = self.h_item(h_i)
        h_u = self.h_user(x_users)
        h = torch.add(h_u, h_i)
        h = self.h_act(h)
        out = self.out_layer(h)
        return self.o_act(out)

    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).to(self.device) \
            .repeat_interleave(self.history_item_id.shape[1], dim=0)
        rating_matrix = torch.zeros(1).to(self.device).repeat(user.shape[0], self.n_items)
        rating_matrix.index_put_((row_indices, col_indices), self.history_item_value[user].flatten())
        return rating_matrix

    def calculate_loss(self, interaction):
        x_users = interaction[self.USER_ID]
        x_items = self.get_rating_matrix(x_users)
        predict = self.forward(x_items, x_users)

        if self.loss_type == 'MSE':
            loss_func = nn.MSELoss(reduction='sum')
        elif self.loss_type == 'BCE':
            loss_func = nn.BCELoss(reduction='sum')
        else:
            raise ValueError('Invalid loss_type, loss_type must in [MSE, BCE]')

        loss = loss_func(predict, x_items)
        # l1-regularization
        loss += self.reg_weight_1 * (self.h_user.weight.norm(p=1) + self.h_item.weight.norm(p=1))
        # l2-regularization
        loss += self.reg_weight_2 * (self.h_user.weight.norm() + self.h_item.weight.norm())

        return loss

    def predict(self, interaction):
        users = interaction[self.USER_ID]
        predict_items = interaction[self.ITEM_ID]

        items = self.get_rating_matrix(users)
        scores = self.forward(items, users)

        return scores[[torch.arange(len(predict_items)).to(self.device), predict_items]]

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]

        items = self.get_rating_matrix(users)
        predict = self.forward(items, users)
        return predict.view(-1)

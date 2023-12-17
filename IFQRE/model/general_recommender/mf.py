from lib2to3.pgen2.token import RPAR
from turtle import forward
from numpy import argmax
import random
import torch
import torch.nn as nn

from IFQRE.model.abstract_recommender import GeneralRecommender
from IFQRE.model.init import xavier_normal_initialization
from IFQRE.utils import InputType

import torch.nn.functional as F


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


class MF(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MF, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.LABEL = config['LABEL_FIELD']
        self.distribution_type = config['distribution_type']
        self.action_type = config['action_type']
        self.unwillingness_type = config['unwillingness_type']
        # self.max_item = config['max_item']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
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

    # def update_distribution(self, u, z):
    #     del_a = torch.zeros(self.model.action_len[u])
    #     for o in range(self.model.action_len[u]):
    #         del_a[o] = -z /self.distribution[u][o]
    #     d=del_a-self.distribution[u]
    #     loss=torch.sum(del_a,self.distribution[u])+learning_rate*torch.sum(pow(d,2))
    #     loss.back
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    def get_unwillingness(self):
        return self.unwillingness

    def get_user_embedding(self, user):
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)

        return user_e, item_e

    def calculate_loss_by_one(self, user, item, label):
        user_e, item_e = self.forward(user, item)
        mf_pred = torch.mul(user_e, item_e)
        res = torch.sum(mf_pred, dim=1)
        output = self.sigmoid(res)
        loss = self.loss(output, label)
        return loss

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        user_e, item_e = self.forward(user, item)
        mf_pred = torch.mul(user_e, item_e)
        res = torch.sum(mf_pred, dim=1)
        output = self.sigmoid(res)
        loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return self.sigmoid(torch.mul(user_e, item_e).sum(dim=1))

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

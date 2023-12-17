# -*- coding: utf-8 -*-
# @Time   : 2020/9/21
# @Author : Zhichao Feng
# @Email  : fzcbupt@gmail.com

# UPDATE
# @Time   : 2020/10/21
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

r"""
DIN
##############################################
Reference:
    Guorui Zhou et al. "Deep Interest Network for Click-Through Rate Prediction" in ACM SIGKDD 2018

Reference code:
    - https://github.com/zhougr1993/DeepInterestNetwork/tree/master/din
    - https://github.com/shenweichen/DeepCTR-Torch/tree/master/deepctr_torch/models

"""

import torch
import random
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MLPLayers, SequenceAttLayer, ContextSeqEmbLayer
from recbole.utils import InputType, FeatureType

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
        self.gamma = torch.rand(self.n_users, dtype=torch.float32, requires_grad=False)
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
                    self.distribution=1 / self.action_len[i]

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
                self.action_num[i] = act
        return self.action,id

    def calculate_loss(self, u, lr, del_a, d):
        loss = torch.sum(del_a * d) + (1 / lr) * torch.sum(pow(del_a - d, 2))
        return loss

class DIN(SequentialRecommender):
    """Deep Interest Network utilizes the attention mechanism to get the weight of each user's behavior according
    to the target items, and finally gets the user representation.

    Note:
        In the official source code, unlike the paper, user features and context features are not input into DNN.
        We just migrated and changed the official source code.
        But You can get user features embedding from user_feat_list.
        Besides, in order to compare with other models, we use AUC instead of GAUC to evaluate the model.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(GameDIN, self).__init__(config, dataset)

        # get field names and parameter value from config
        self.LABEL_FIELD = config['LABEL_FIELD']
        self.LABEL = config['LABEL_FIELD']
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.device = config['device']
        self.pooling_mode = config['pooling_mode']
        self.dropout_prob = config['dropout_prob']

        self.types = ['user', 'item']
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()
        self.n_users=self.user_feat.length
        # init MLP layers
        # self.dnn_list = [(3 * self.num_feature_field['item'] + self.num_feature_field['user'])
        #                  * self.embedding_size] + self.mlp_hidden_size
        num_item_feature = sum(
            1 if dataset.field2type[field] != FeatureType.FLOAT_SEQ else dataset.num(field)
            for field in self.item_feat.interaction.keys()
        )
        self.dnn_list = [3 * num_item_feature * self.embedding_size] + self.mlp_hidden_size
        self.att_list = [4 * num_item_feature * self.embedding_size] + self.mlp_hidden_size

        mask_mat = torch.arange(self.max_seq_length).to(self.device).view(1, -1)  # init mask
        self.attention = SequenceAttLayer(
            mask_mat, self.att_list, activation='Sigmoid', softmax_stag=False, return_seq_weight=False
        )
        self.dnn_mlp_layers = MLPLayers(self.dnn_list, activation='Dice', dropout=self.dropout_prob, bn=True)

        self.embedding_layer = ContextSeqEmbLayer(dataset, self.embedding_size, self.pooling_mode, self.device)
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['embedding_layer']


        self.distribution_type = config['distribution_type']
        self.action_type = config['action_type']
        self.unwillingness_type = config['unwillingness_type']
        self.history, self.history_value, self.history_len = dataset.history_item_matrix()
        self.max_item = torch.max(self.history_len).item()
        self.max_action = pow(2, self.max_item)
        self.strategy_mask = torch.zeros(self.n_users, self.n_items)
        self.pow_mask = torch.zeros(self.n_users, self.n_items)
        self.action_len = pow(2, self.history_len)
        self.unwillingness = torch.zeros(self.n_users, self.n_items)
        self.init_unwillingness()
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
    def init_unwillingness(self):
        if self.unwillingness_type == 1:
            for i in range(self.n_users):
                self.unwillingness[i] = torch.rand(self.n_items)

        elif self.unwillingness_type == 2:
            for i in range(self.n_users):
                if self.history_len[i] > 0:
                    self.unwillingness[i] = torch.ones(self.n_items)

        elif self.unwillingness_type == 3:
            for i in range(self.n_users):
                count = torch.zeros(self.n_items)
                for k in range(self.history_len[i]):
                    count[self.history_len[k].item()] += 1
                for k in range(self.history_len[i]):
                    self.unwillingness[i] = 1 / count[k]

        return self.unwillingness*self.strategy_mask

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, user, item_seq, item_seq_len, next_items):

        max_length = item_seq.shape[1]
        # concatenate the history item seq with the target item to get embedding together
        item_seq_next_item = torch.cat((item_seq, next_items.unsqueeze(1)), dim=-1)
        sparse_embedding, dense_embedding = self.embedding_layer(user, item_seq_next_item)
        # concat the sparse embedding and float embedding
        feature_table = {}
        for type in self.types:
            feature_table[type] = []
            if sparse_embedding[type] is not None:
                feature_table[type].append(sparse_embedding[type])
            if dense_embedding[type] is not None:
                feature_table[type].append(dense_embedding[type])

            feature_table[type] = torch.cat(feature_table[type], dim=-2)
            table_shape = feature_table[type].shape
            feat_num, embedding_size = table_shape[-2], table_shape[-1]
            feature_table[type] = feature_table[type].view(table_shape[:-2] + (feat_num * embedding_size,))

        user_feat_list = feature_table['user']
        item_feat_list, target_item_feat_emb = feature_table['item'].split([max_length, 1], dim=1)
        target_item_feat_emb = target_item_feat_emb.squeeze(1)

        # attention
        user_emb = self.attention(target_item_feat_emb, item_feat_list, item_seq_len)
        user_emb = user_emb.squeeze(1)

        # input the DNN to get the prediction score
        din_in = torch.cat([user_emb, target_item_feat_emb, user_emb * target_item_feat_emb], dim=-1)
        din_out = self.dnn_mlp_layers(din_in)
        preds = self.dnn_predict_layers(din_out)
        preds = self.sigmoid(preds)

        return preds.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL_FIELD]
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        output = self.forward(user, item_seq, item_seq_len, next_items)
        loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        scores = self.forward(user, item_seq, item_seq_len, next_items)
        return scores

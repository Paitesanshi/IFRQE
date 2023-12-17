import torch
import torch.nn as nn
from torch.nn.init import normal_
import random
from IFQRE.model.abstract_recommender import GeneralRecommender
from IFQRE.model.layers import MLPLayers
from IFQRE.utils import InputType


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
                    self.distribution[i].fill_(1 / self.action_len[i])


    def get_action_by_distribution(self):
        self.action.fill_(0)
        self.action_num=list()
        if self.action_type == 1:
            for i in range(self.n_users):
                o = random.randint(0, self.action_len[i] - 1)
                self.action_num.append(o)
                for j in range(self.history_len[i]):
                    self.action[i, self.history[i][j].item()] = (o.item() >> j) & 1
        elif self.action_type == 2:
            for i in range(self.n_users):
                o = torch.argmax(self.distribution[i])
                self.action_num.append(o)
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
        return self.action,self.action_num


class NeuMF(GeneralRecommender):
    r"""NeuMF is an neural network enhanced matrix factorization model.
    It replace the dot product to mlp for a more precise user-item interaction.

    Note:

        Our implementation only contains a rough pretraining function.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.mf_embedding_size = config['mf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.mf_train = config['mf_train']
        self.mlp_train = config['mlp_train']
        self.use_pretrain = config['use_pretrain']
        self.mf_pretrain_path = config['mf_pretrain_path']
        self.mlp_pretrain_path = config['mlp_pretrain_path']
        self.distribution_type = config['distribution_type']
        self.action_type = config['action_type']
        self.unwillingness_type = config['unwillingness_type']

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        self.mlp_layers = MLPLayers([2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob)
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_size[-1], 1)
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)
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

    def get_random_action(self, threshold):
        o = torch.rand(self.n_users, self.n_items)
        o = o * self.strategy_mask
        o = o > threshold
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

    def get_action(self):
        return self.distribution.predict()

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
                    cates=dataset.item_feat.interaction['genres'][self.history[i][k].item()]
                    cates=torch.unique(cates)
                    for j in range(len(cates)):
                        # if cates[j].item()==0:
                        #     continue
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
                        cates = dataset.item_feat.interaction['genres'][self.history[i][k].item()]
                        cates = torch.unique(cates)
                        for j in range(len(cates)):
                            # if cates[j].item()==0:
                            #     continue
                            if categories[cates[j].item()]>maxk:
                                maxk=categories[cates[j].item()]
                        # print("error",maxk,maxv,minv)
                        # if maxk==0 or maxv==0 or minv==0:
                        #     print("error",maxk,maxv,minv)
                        #     print("cates: ",cates)
                        self.unwillingness[i,self.history[i][k].item()] =(1/maxk-1/maxv)/ (1/minv-1/maxv)/2+0.5
        return self.unwillingness * self.strategy_mask

    def get_unwillingness(self):
        return self.unwillingness


    def load_pretrain(self):
        r"""A simple implementation of loading pretrained parameters.

        """
        mf = torch.load(self.mf_pretrain_path)
        mlp = torch.load(self.mlp_pretrain_path)
        self.user_mf_embedding.weight.data.copy_(mf.user_mf_embedding.weight)
        self.item_mf_embedding.weight.data.copy_(mf.item_mf_embedding.weight)
        self.user_mlp_embedding.weight.data.copy_(mlp.user_mlp_embedding.weight)
        self.item_mlp_embedding.weight.data.copy_(mlp.item_mlp_embedding.weight)

        for (m1, m2) in zip(self.mlp_layers.mlp_layers, mlp.mlp_layers.mlp_layers):
            if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                m1.weight.data.copy_(m2.weight)
                m1.bias.data.copy_(m2.bias)

        predict_weight = torch.cat([mf.predict_layer.weight, mlp.predict_layer.weight], dim=1)
        predict_bias = mf.predict_layer.bias + mlp.predict_layer.bias

        self.predict_layer.weight.data.copy_(0.5 * predict_weight)
        self.predict_layer.weight.data.copy_(0.5 * predict_bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        elif self.mf_train:
            output = self.sigmoid(self.predict_layer(mf_output))
        elif self.mlp_train:
            output = self.sigmoid(self.predict_layer(mlp_output))
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain.

        """
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)

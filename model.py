import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, hyps, train_ds, device='cpu'):
        super().__init__()
        self.embed_size = hyps.embedding_size
        self.regularize_weight = hyps.regularize_weight
        self.device = device
        self.user_num = train_ds.num_user
        self.item_num = train_ds.num_item

        if hyps.pretrain_embed:
            load_data = torch.load(hyps.pretrain_path, map_location='cpu')
            self.item_embedding = nn.Parameter(F.normalize(load_data['item_embedding']).to(self.device))
            self.user_embedding = nn.Parameter(F.normalize(load_data['user_embedding']).to(self.device))
            # self.item_embedding = load_data['item_embedding'].to(self.device)
            # self.user_embedding = load_data['user_embedding'].to(self.device)
        else:
            self.item_embedding = nn.Parameter(torch.FloatTensor(self.item_num, self.embed_size))
            self.user_embedding = nn.Parameter(torch.FloatTensor(self.user_num, self.embed_size))
            nn.init.xavier_normal_(self.item_embedding)
            nn.init.xavier_normal_(self.user_embedding)

    def propagate(self, *args, **kwargs):
        """
        Perform item-to-user, user-to-item or item-to-item embedding
        :return: embedding
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """
        Predict score given user and item embedding
        :return: single score
        """
        raise NotImplementedError

    def regularize(self, user_embeddings, item_embeddings):
        """
        regularize loss for user embedding and item embedding
        :return: single float
        """
        return self.regularize_weight * ((user_embeddings ** 2).sum() + (item_embeddings ** 2).sum())

    def forward(self, users, items):
        """
        forward propagation to get recommendation output for users with given items
        :return: score matrix
        """
        raise NotImplementedError

    def evaluate(self, users):
        """
        Use for testing, return score of all items for given users
        """
        raise NotImplementedError


# MF Recommender system
class MF(Embedding):
    def __init__(self, hyps, train_ds, device):
        super().__init__(hyps, train_ds, device)

    def propagate(self):
        return self.user_embedding, self.item_embedding

    def predict(self, user_embedding, item_embedding):
        # size = (batch_size, 2, embedding_size). 2 => pos and neg sampling
        return torch.sum(user_embedding * item_embedding, dim=2)

    def forward(self, user, item):
        # propagate to get user and item embedding
        users_feature, item_feature = self.propagate()

        # get embedding of several items
        item_embeddings = item_feature[item]

        # get embedding of several users => expand 2 dimention to sampling size (usually for pos and neg sampling)
        user_embeddings = users_feature[user].expand(-1, item.shape[1], -1)

        pred = self.predict(user_embeddings, item_embeddings)
        L2_loss = self.regularize(user_embeddings, item_embeddings)

        return pred, L2_loss

    def evaluate(self, users):
        # propagate to get user and item embedding
        users_feature, item_feature = self.propagate()

        # get user embedding of several users
        user_feature = users_feature[users]

        # shape = (num_user, num_item)
        scores = torch.mm(user_feature, item_feature.t())
        return scores


# Graph-based recommender system
class Multi_Behavior_Graph_Base(Embedding):
    def __init__(self, hyps, train_ds, device, eps=1e-8):
        super().__init__(hyps, train_ds, device)

        # Adding value to avoid zero dividing
        self.eps = eps
        # Behaviors interaction dict. Each key is one behavior with value is a matrix ==> size (num_user, num_item)
        self.adj_matrix_dict = train_ds.adj_matrix_dict
        # Weight for each behavior => use to calculate item-to-user embedding ==> size (num_behavior, )
        self.behavior_weight = hyps.behavior_weight
        # item-item graph for each behavior ==> size (num_item, num_item)
        self.item_graph = train_ds.item_graph
        # Training matrix (all behavior interaction) use for extract item feature ==> size (num_user, num_item)
        self.train_matrix = train_ds.train_matrix.to(self.device)
        # List of behavior ==> size (num_behavior,)
        self.behaviors = train_ds.behaviors
        # weight for 2 output score ==> float
        self.sigmoid_weight = hyps.sigmoid_weight
        # Degree matrix for item-item graph ==> size (num_item, num_behavior)
        self.item_graph_degree = train_ds.item_graph_degree
        # Degree matrix for users ==> size (num_user, num_behavior)
        self.user_behaviour_degree = train_ds.user_behaviour_degree.to(self.device)
        # Message dropout or forward dropout
        self.message_drop = nn.Dropout(p=hyps.message_dropout)
        # Dropout some ground true target behavior
        self.gt_drop = nn.Dropout(p=hyps.node_dropout)
        # Dropout some user-item link when training
        self.node_drop = nn.ModuleList([nn.Dropout(p=hyps.node_dropout) for _ in self.adj_matrix_dict])

        # Setting parameter and put it into corresponding device
        self.__to_device()
        self.__param_init()

    def __to_device(self):
        for key in self.adj_matrix_dict:
            self.adj_matrix_dict[key] = self.adj_matrix_dict[key].to(self.device)
        for key in self.item_graph:
            self.item_graph[key] = self.item_graph[key].to(self.device)
        for key in self.item_graph_degree:
            self.item_graph_degree[key] = self.item_graph_degree[key].to(self.device)

    def __param_init(self):
        # Weight for user-item score and item-item score
        self.sigmoid_W = nn.Parameter(torch.ones(self.embed_size * 2, 1)*self.sigmoid_weight)

        # Weight for each behavior
        self.behavior_weight = nn.Parameter(torch.FloatTensor(self.behavior_weight))

        # Encoding matrix for item-to-item score ==> 2 *embed
        self.item_behaviour_W = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.embed_size * 2, self.embed_size * 2)) for _ in self.behavior_weight])
        for param in self.item_behaviour_W:
            nn.init.xavier_normal_(param)

        # Encoding matrix for item-to-item propagation
        self.item_propagate_W = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size)) for _ in self.behavior_weight])
        for param in self.item_propagate_W:
            nn.init.xavier_normal_(param)

        # GCN Embedding encoding matrix
        self.W = nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size))
        nn.init.xavier_normal_(self.W)

        # # Residual for user and item feature
        self.user_W = nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size * 2))
        self.item_W = nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size * 2))
        nn.init.xavier_normal_(self.user_W)
        nn.init.xavier_normal_(self.item_W)

    def forward(self, user, item):
        # node dropout on train matrix (ground true user-item purchase)
        indices = self.train_matrix._indices()
        values = self.train_matrix._values()
        values = self.gt_drop(values)
        train_matrix = torch.sparse_coo_tensor(indices, values, size=self.train_matrix.shape)

        # Calculate users weight for each behavior
        weight = self.behavior_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        user_behaviour_weight = self.user_behaviour_degree * self.behavior_weight.unsqueeze(0) / (
                    total_weight + self.eps)

        user_feature = None
        score2 = 0
        for i, key in enumerate(self.adj_matrix_dict):
            # node dropout - drop some interaction between user and item
            indices = self.adj_matrix_dict[key]._indices()
            values = self.adj_matrix_dict[key]._values()
            values = self.node_drop[i](values)
            behavior_matrix_drop = torch.sparse_coo_tensor(indices, values, size=self.adj_matrix_dict[key].shape)

            # GCN for item-item co-interaction by all users
            item_item_propagation = torch.mm(
                torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + self.eps),
                self.item_propagate_W[i])
            # Concat item-item embedding with original embedding to avoid gradient vanishing
            item_item_propagation = torch.cat((self.item_embedding, item_item_propagation), dim=1)
            # Taking particular item to training
            used_item_item_embedding = item_item_propagation[item]

            # GCN for item-to-user propagation => get user feature through item which user interact with
            user_neighbor_feature = torch.mm(behavior_matrix_drop, self.item_embedding) / (
                    self.user_behaviour_degree[:, i].unsqueeze(-1) + self.eps)

            # GCN for item-to-user propagation2
            neigborItem_to_user_feature = torch.mm(behavior_matrix_drop, item_item_propagation) / (
                    self.user_behaviour_degree[:, i].unsqueeze(-1) + self.eps)

            # Initialize feature and score
            if i == 0:
                # Item-to-user feature for multi-behavior weight
                user_feature = user_behaviour_weight[:, i].unsqueeze(-1) * user_neighbor_feature

                # Encoding for item co-interact => item propagation for user
                item_to_user_cointeract_feature = torch.mm(neigborItem_to_user_feature, self.item_behaviour_W[i])
                # Taking particular item propagation for user
                used_item_to_user_cointeract_feature = item_to_user_cointeract_feature[user].expand(-1, item.shape[1], -1)

                # Calculate ranking score
                score2 = torch.sum(used_item_to_user_cointeract_feature * used_item_item_embedding, dim=2)

            # Aggregated for multi-behavior
            else:
                user_feature += user_behaviour_weight[:, i].unsqueeze(-1) * user_neighbor_feature
                item_to_user_cointeract_feature = torch.mm(neigborItem_to_user_feature, self.item_behaviour_W[i])
                used_item_to_user_cointeract_feature = item_to_user_cointeract_feature[user].expand(-1, item.shape[1], -1)
                score2 += torch.sum(used_item_to_user_cointeract_feature * used_item_item_embedding, dim=2)

        # Behavior normalize for score2
        score2 = score2 / len(self.behavior_weight)

        # Use-to-item propagation
        item_feature = torch.mm(train_matrix.t(), self.user_embedding)

        # Encoding feature
        residual_user_feature = torch.mm(user_feature, self.user_W)
        residual_item_feature = torch.mm(item_feature, self.item_W)
        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)

        # Concat feature with original embedding to avoid gradient vanishing
        user_feature = torch.cat((self.user_embedding, user_feature), dim=1)
        item_feature = torch.cat((self.item_embedding, item_feature), dim=1)
        user_feature += residual_user_feature
        item_feature += residual_item_feature

        # message dropout
        user_feature = self.message_drop(user_feature)
        item_feature = self.message_drop(item_feature)

        # Getting feature corresponding to in process user and item
        used_user_feature = user_feature[user].expand(-1, item.shape[1], -1)
        used_item_feature = item_feature[item]
        score1 = torch.sum(used_user_feature * used_item_feature, dim=2)

        # Final ranking score
        # scores = score1 + self.sigmoid_weight * score2
        score_weight = torch.mm(user_feature, self.sigmoid_W)[user].squeeze(dim=2).expand(-1, item.shape[1])
        scores = score1 * torch.sigmoid(score_weight) + score2 * (1 - torch.sigmoid(score_weight))

        # Regularization loss
        L2_loss = self.regularize(used_user_feature, used_item_feature)

        return scores, L2_loss

    def evaluate(self, users):
        # Calculate users weight for each behavior
        weight = self.behavior_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        user_behaviour_weight = self.user_behaviour_degree * self.behavior_weight.unsqueeze(0) / (
                    total_weight + self.eps)

        user_feature = None
        score2 = 0
        for i, key in enumerate(self.adj_matrix_dict):
            # GCN for item-item co-interaction by all users
            item_item_propagation = torch.mm(
                torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + self.eps),
                self.item_propagate_W[i])
            # Concat item-item embedding with original embedding to avoid gradient vanishing
            item_item_propagation = torch.cat((self.item_embedding, item_item_propagation), dim=1)

            # GCN for item-to-user propagation => get user feature through item which user interact with
            user_neighbor_feature = torch.mm(self.adj_matrix_dict[key], self.item_embedding) / (
                    self.user_behaviour_degree[:, i].unsqueeze(-1) + self.eps)

            # GCN for item-to-user propagation2
            neigborItem_to_user_feature = torch.mm(self.adj_matrix_dict[key], item_item_propagation) / (
                    self.user_behaviour_degree[:, i].unsqueeze(-1) + self.eps)

            # Initialize feature and score
            if i == 0:
                # Item-to-user feature for multi-behavior weight
                user_feature = user_behaviour_weight[:, i].unsqueeze(-1) * user_neighbor_feature

                # Encoding for item co-interact => item propagation for user
                item_to_user_cointeract_feature = torch.mm(neigborItem_to_user_feature, self.item_behaviour_W[i])
                # Taking particular item propagation for user
                used_item_to_user_cointeract_feature = item_to_user_cointeract_feature[users]

                # Calculate ranking score
                score2 = torch.mm(used_item_to_user_cointeract_feature, item_item_propagation.t())

            # Aggregated for multi-behavior
            else:
                user_feature += user_behaviour_weight[:, i].unsqueeze(-1) * user_neighbor_feature
                item_to_user_cointeract_feature = torch.mm(neigborItem_to_user_feature, self.item_behaviour_W[i])
                used_item_to_user_cointeract_feature = item_to_user_cointeract_feature[users]
                score2 += torch.mm(used_item_to_user_cointeract_feature, item_item_propagation.t())

        # Behavior normalize for score2
        score2 = score2 / len(self.behavior_weight)

        # Use-to-item propagation
        item_feature = torch.mm(self.train_matrix.t(), self.user_embedding)

        # Encoding feature
        residual_user_feature = torch.mm(user_feature, self.user_W)
        residual_item_feature = torch.mm(item_feature, self.item_W)
        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)

        # Concat feature with original embedding to avoid gradient vanishing
        user_feature = torch.cat((self.user_embedding, user_feature), dim=1)
        item_feature = torch.cat((self.item_embedding, item_feature), dim=1)
        user_feature += residual_user_feature
        item_feature += residual_item_feature

        tmp_user_feature = user_feature[users]
        score1 = torch.mm(tmp_user_feature, item_feature.t())

        # Final ranking score for a user
        # scores = score1 + self.sigmoid_weight * score2
        score_weight = torch.mm(user_feature, self.sigmoid_W)[users]
        scores = score1 * torch.sigmoid(score_weight) + score2 * (1 - torch.sigmoid(score_weight))

        return scores

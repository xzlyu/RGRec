import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve

device = "cuda" if torch.cuda.is_available() else "cpu"


class RKGCN(nn.Module):
    def __init__(self, args, e_num, r_num, rule_size, adj_e_id_by_r_id_dict):
        super(RKGCN, self).__init__()
        self.e_num = e_num
        self.r_num = r_num
        self.rule_size = rule_size
        self.adj_e_id_by_r_id_dict = adj_e_id_by_r_id_dict
        self._build_param(args)
        self._build_model()

    def _build_param(self, args):
        self.dim = args.rkgcn_dim
        self.batch_size = args.rkgcn_batch_size
        self.max_step = args.rkgcn_max_step
        self.neighbour_size = args.rkgcn_neighbour_size
        self.dropout = args.rkgcn_dropout

    def _build_model(self):
        self.ent_embed = nn.Embedding(self.e_num, self.dim).to(device)
        self.rule_embed = nn.Embedding(1, self.rule_size).to(device)
        # self.rule_embed = torch.Tensor(np.ones([1, self.rule_size]) / self.rule_size).to(device)
        self.aggregate_layer = nn.Linear(self.dim, self.dim).to(device)
        torch.nn.init.xavier_uniform(self.ent_embed.weight)
        torch.nn.init.xavier_uniform(self.rule_embed.weight)
        torch.nn.init.xavier_uniform(self.aggregate_layer.weight)

    def user_rep(self, user_array, rule_list):
        adj_e_list = []

        for one_rule in rule_list:
            # entities, list, [i,[batch_size,neighbour_size^i]], i in [1,...,max_step]
            adj_e = self.get_adj_ent_of_r_id(one_rule, user_array)
            adj_e_list.append(adj_e)

        res_list = []
        for step_i in range(self.args.max_step + 1):
            one_step_node = []
            for rule_idx in range(len(rule_list)):
                one_step_node.append(adj_e_list[rule_idx][step_i])
            # np.hstack(one_step_node), [batch_size,rule_num * neighbour^step_i]
            res_list.append(np.hstack(one_step_node))

        # res, [batch_size, rule_num, dim]
        res = self.aggregate(res_list)

        # output, [batch_size,]
        output = torch.sum(
            res.to(device) * self.rule_embed(torch.LongTensor([0]).to(device)).view((self.rule_size, 1)).to(device),
            dim=-2)

        return output

    def update(self, user_item_array, rule_list, label_array):

        output = self.user_rep(user_item_array[:, 0], rule_list)
        item_list = np.array(user_item_array)[:, 1]
        item_embed = self.ent_embed(torch.LongTensor(item_list).to(device))
        res_prob = torch.sigmoid(torch.sum(output * item_embed, dim=-1)).to(device)

        criterion = nn.BCELoss().to(device)

        loss = criterion(res_prob, torch.Tensor(label_array).to(device))
        loss.backward()
        return loss.cpu().item() / self.batch_size

    def get_adj_ent_of_r_id(self, one_rule, seed):
        # seed, np array, [batch_size,1]
        # entities, list, [np array], [[batch_size,1]]
        entities = [seed]
        for idx in range(self.max_step):
            if idx >= len(one_rule):
                r_id = self.r_num
            else:
                r_id = one_rule[idx]
            # neighbours, np array, [e_num, neighbour_size]
            neighbours = self.adj_e_id_by_r_id_dict[r_id]
            # r_neighbour, np array, [self.batch_size,-1]
            # entities[idx], np_array, [batch_size, neighbour_size^idx]
            r_neighbour = neighbours[entities[idx], :].reshape([self.batch_size, -1])
            entities.append(r_neighbour)
        # entities, list, [i,[batch_size,neighbour_size^i]], i in [0,1,...,max_step]
        return entities

    def mix_neighbour_vectors(self, neighbour_vectors):
        # neighbor_vectors, [batch_size, -1, neighbor_size, dim]
        # [batch_size, -1, dim]
        neighbors_aggregated = torch.mean(neighbour_vectors, dim=2)

        return neighbors_aggregated

    # self_vectors, [batch_size,-1,dim]
    # neighbor_vectors, [batch_size,-1,neighbour_size,dim]
    def sum_aggreator(self, self_vectors, neighbor_vectors, act):
        # [batch_size, -1, dim]
        neighbors_agg = self.mix_neighbour_vectors(neighbor_vectors)

        # [-1, dim]
        output = (self_vectors + neighbors_agg).view([-1, self.dim])
        output = F.dropout(output, p=self.dropout)
        output = self.aggregate_layer(output)

        # [batch_size, -1, dim]
        output = output.view([self.batch_size, -1, self.dim]).to(device)

        return act(output)

    def aggregate(self, entities):

        entity_vectors = [self.ent_embed(torch.LongTensor(i).to(device)).to(device) for i in entities]

        for i in range(self.max_step):
            act = torch.tanh if i == self.max_step - 1 else torch.relu
            entity_vectors_next_iter = []
            for hop in range(self.max_step - i):
                shape = [self.args.batch_size, -1, self.neighbour_size, self.dim]
                vector = self.sum_aggreator(
                    self_vectors=entity_vectors[hop].view([self.batch_size, -1, self.dim]),
                    neighbor_vectors=entity_vectors[hop + 1].view(shape), act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = entity_vectors[0].view([self.batch_size, -1, self.dim])
        return res

    def auc_f1_prec_recal_scores_eval(self, user_item_array, rule_list, label_array):
        output = self.user_rep(user_item_array[:, 0], rule_list)
        item_list = np.array(user_item_array)[:, 1]
        item_embed = self.ent_embed(torch.LongTensor(item_list).to(device))
        scores = torch.sigmoid(torch.sum(output * item_embed, dim=-1)).cpu().detach().numpy()

        auc = roc_auc_score(y_true=label_array, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0

        precision = precision_score(y_true=label_array, y_pred=scores)
        recall = recall_score(y_true=label_array, y_pred=scores)
        f1 = f1_score(y_true=label_array, y_pred=scores)
        return auc, f1, precision, recall, scores

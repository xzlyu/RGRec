import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve

from src.Args import args
from src.Graph import Graph

device = "cuda" if torch.cuda.is_available() else "cpu"


class PRA(nn.Module):
    def __init__(self, feature_size, lr, l2):
        super(PRA, self).__init__()
        self.feature_size = feature_size
        self.lr = lr
        self.l2 = l2

    def _build_model(self):
        self.layer1 = nn.Linear(self.feature_size, 1)

    def forward(self, input_x):
        return self.layer1(input_x)

    def update(self, train_x, train_y):
        output = self.forward(torch.Tensor(train_x))
        output = nn.sigmoid(output)

        criterion = nn.BCELoss().to(device)
        loss = criterion(output, torch.Tensor(train_y).to(device))
        loss.backward()

        return loss.cpu().item() / len(train_x)

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_model(file_path)


def train_pra_model(pra_args, pra_model, train_data, eval_data, test_data):
    lr = pra_args.pra_lr
    l2_weight = pra_args.pra_l2_weight
    batch_size = pra_args.pra_batch_size
    n_epochs = pra_args.pra_n_epochs

    optimizer = torch.optim.Adam(pra_model.parameters(), lr=lr, weight_decay=l2_weight)

    for epoch_i in range(n_epochs):
        start = 0
        np.random.shuffle(train_data)
        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]

        running_loss = 0
        while start + batch_size < train_data.shape[0]:
            optimizer.zero_grad()
            loss = pra_model.update(train_x[start:start + batch_size], train_y[start:start + batch_size])
            start += batch_size
            running_loss += loss
            optimizer.step()

        running_loss /= start - batch_size

        print("Epoch: {}/{}, Loss: {}"
              .format(epoch_i, n_epochs, running_loss))

        train_auc, train_prec, train_reca, train_f = eval_prec_reca_f(pra_model, train_data)
        eval_auc, eval_prec, eval_reca, eval_f = eval_prec_reca_f(pra_model, eval_data)
        test_auc, test_prec, test_reca, test_f = eval_prec_reca_f(pra_model, test_data)

        print('epoch %d    train auc: %.4f  f1: %.4f prec: %.4f reca: %.4f  '
              'eval auc: %.4f  f1: %.4f prec: %.4f reca: %.4f    '
              'test auc: %.4f  f1: %.4f prec: %.4f reca: %.4f'
              % (epoch_i, train_auc, train_f, train_prec, train_reca,
                 eval_auc, eval_f, eval_prec, eval_reca,
                 test_auc, test_f, test_prec, test_reca))


def eval_prec_reca_f(pra_model, data):
    data_x = data[:, :-1]
    data_y = data[:, -1]

    scores = pra_model.forward(data_x)
    scores[scores >= 0.5] = 1
    scores[scores < 0.5] = 0
    precision = precision_score(y_true=data_y, y_pred=scores)
    recall = recall_score(y_true=data_y, y_pred=scores)
    f1 = f1_score(y_true=data_y, y_pred=scores)

    auc = roc_auc_score(y_true=data_y, y_score=scores)

    return auc, precision, recall, f1

if __name__ == "__main__":
    g = Graph(args)
    g.load_e_r_mapping()
    g.construct_kg()
    g.construct_adj_e_id()

    rule_list = g.get_rules(g.r_name2id['interact'])
    rule_list.sort(key=lambda x: x[0])
    rule_id_list = [rule for _, _, rule, _ in rule_list]

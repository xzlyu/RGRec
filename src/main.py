import argparse

import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from src.Args import args
from src.Graph import Graph
from src.rkgcn_model import RKGCN

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    train_data = np.load(args.train_file)
    eval_data = np.load(args.eval_file)
    test_data = np.load(args.test_file)

    return train_data, eval_data, test_data
    # train_one_num = np.sum(train_data[:, 2])
    # train_zero_num = train_data[:, 2].shape[0] - train_one_num
    #
    # eval_one_num = np.sum(eval_data[:, 2])
    # eval_zero_num = eval_data[:, 2].shape[0] - eval_one_num
    #
    # test_one_num = np.sum(test_data[:, 2])
    # test_zero_num = test_data[:, 2].shape[0] - test_one_num
    #
    # print("Train: 1/0 {}/{}, Eval: 1/0 {}/{}, Test: 1/0 {}/{}".format(train_one_num, train_zero_num, eval_one_num,
    #                                                                   eval_zero_num, test_one_num, test_zero_num))

    # return train_data[:, 0:2], eval_data[:, 0:2], test_data[:, 0:2], train_data[:, 2], eval_data[:, 2], test_data[:, 2]


def train_rkgcn(rkgcn_model, rule_id_list):
    lr = args.rkgcn_lr
    l2_weight = args.rkgcn_l2_weight
    n_epochs = args.rkgcn_n_epochs
    batch_size = args.rkgcn_batch_size

    train_data_label, eval_data_label, test_data_label = load_data()

    optimizer = torch.optim.Adam(rkgcn_model.parameters(), lr=lr, weight_decay=l2_weight)

    eval_auc_list = []

    for epoch_i in range(n_epochs):

        np.random.shuffle(train_data_label)
        start = 0
        # skip the last incomplete minibatch if its size < batch size
        while start + batch_size <= train_data_label.shape[0]:
            optimizer.zero_grad()
            loss = rkgcn_model.update(train_data_label[start:start + args.rkgcn_batch_size, 0:2], rule_id_list,
                                      train_data_label[start:start + args.rkgcn_batch_size, 2])
            start += batch_size
            # print("Epoch: {}/{}, Start: {}/{}, Loss: {}"
            #       .format(epoch_i, args.n_epochs, start, train_data.shape[0], loss))
            optimizer.step()

        if (epoch_i + 1) % 5 != 0:
            continue

        train_auc, train_f1, train_prec, train_reca = ctr_eval(epoch_i, rkgcn_model,
                                                               train_data_label[:, 0:2],
                                                               train_data_label[:, 2],
                                                               rule_id_list,
                                                               args.rkgcn_batch_size)
        eval_auc, eval_f1, eval_prec, eval_reca = ctr_eval(epoch_i, rkgcn_model,
                                                           eval_data_label[:, 0:2],
                                                           eval_data_label[:, 2],
                                                           rule_id_list,
                                                           args.rkgcn_batch_size)
        test_auc, test_f1, test_prec, test_reca = ctr_eval(epoch_i, rkgcn_model,
                                                           test_data_label[:, 0:2],
                                                           test_data_label[:, 2],
                                                           rule_id_list,
                                                           args.rkgcn_batch_size)
        eval_auc_list.append(eval_auc)

        print(
            'epoch %d    train auc: %.4f  f1: %.4f prec: %.4f reca: %.4f  eval auc: %.4f  f1: %.4f prec: %.4f reca: %.4f    test auc: %.4f  f1: %.4f prec: %.4f reca: %.4f'
            % (epoch_i, train_auc, train_f1, train_prec, train_reca, eval_auc, eval_f1, eval_prec, eval_reca, test_auc,
               test_f1, test_prec, test_reca))

        if len(eval_auc_list) == 1 or eval_auc_list[-1] > eval_auc_list[-2]:
            rkgcn_model.save_model()

        if len(eval_auc_list) != 1 and eval_auc_list[-1] <= eval_auc_list[-2]:
            break


def ctr_eval(epoch_i, model, data, label, rule_id_list, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    score_list = []
    label_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1, prec, reca, score = model.auc_f1_prec_recal_scores_eval(data[start:start + batch_size], rule_id_list,
                                                                         label[start:start + batch_size])
        auc_list.append(auc)
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(reca)
        label_list.extend(list(label[start:start + batch_size]))
        score_list.extend(list(score))

        start += batch_size

    fpr, tpr, thresholds = roc_curve(y_true=label_list, y_score=score_list, pos_label=1)

    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    # plt.show()
    plt.savefig('../data/music/pic/{}.png'.format(epoch_i))
    return float(np.mean(auc_list)), float(np.mean(f1_list)), float(np.mean(precision_list)), float(
        np.mean(recall_list))


if __name__ == "__main__":
    g = Graph(args)
    g.load_e_r_mapping()
    g.construct_kg()
    g.construct_adj_e_id()

    rule_list = g.get_rules(g.r_name2id['interact'])
    rule_list.sort(key=lambda x: x[0])
    rule_id_list = [rule for _, _, rule, _ in rule_list]

    rkgcn = RKGCN(args, len(g.e_id2name), len(g.r_id2name), len(rule_id_list), g.adj_e_id_by_r_id).to(device)
    train_rkgcn(rkgcn, rule_id_list)

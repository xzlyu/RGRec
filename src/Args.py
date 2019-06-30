import argparse

DATASET = "music"

parser = argparse.ArgumentParser()

parser.add_argument('--rule_seg', type=str, default="->",
                    help='the signal to split r_name of a rule')

parser.add_argument('--kg_file', type=str, default="../data/" + DATASET + "/inv_kg_final",
                    help="file to store the whole kg.")
parser.add_argument('--entity_file', type=str, default="../data/" + DATASET + "/entity_name2id.txt",
                    help="file to store entity and entity id.")
parser.add_argument('--relation_file', type=str, default="../data/" + DATASET + "/relation_name2id.txt",
                    help="file to store relation and relation id.")
parser.add_argument('--rule_file', type=str, default="../data/" + DATASET + "/model/rule.txt",
                    help="file to store the rule searched.")

parser.add_argument('--train_file', type=str, default="../data/" + DATASET + "/model/train.npy",
                    help="file to store train data")
parser.add_argument('--eval_file', type=str, default="../data/" + DATASET + "/model/eval.npy",
                    help="file to store eval data.")
parser.add_argument('--test_file', type=str, default="../data/" + DATASET + "/model/test.npy",
                    help="file to store test data.")
parser.add_argument('--converted_rating_file', type=str, default="../data/" + DATASET + '/converted_ratings_final.npy',
                    help="file to store converted rating.")

parser.add_argument('--max_step', type=int, default=3, help="max length of rules")
parser.add_argument('--sampled_ht_ratio', type=float, default=0.8,
                    help='the ratio of [h_id,t_id] of r_id used to search inference rules')
parser.add_argument("--chi_thresh", type=float, default=0.01, help="threshold for chi-square")
parser.add_argument('--filter_inv_pattern', type=bool, default=False,
                    help='remove rules with inverse and neighbouring relations')
parser.add_argument('--reserved_rule_num_by_frequency', type=int, default=10000,
                    help="the number of reserved rules by descending frequency")

parser.add_argument('--neighbour_size', type=int, default=3,
                    help="the num of neighbourhood to sample of an entity")

# rule gcn training params
parser.add_argument('--rkgcn_dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--rkgcn_l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--rkgcn_lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--rkgcn_batch_size', type=int, default=256, help='batch size')
parser.add_argument('--rkgcn_n_epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--rkgcn_dropout', type=float, default=0, help='probility to drop out')

# parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')


# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--ls_weight', type=float, default=0, help='weight of LS regularization')
# parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# pra training params
parser.add_argument('--pra_l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--pra_lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--pra_batch_size', type=int, default=256, help='batch size')
parser.add_argument('--pra_n_epochs', type=int, default=100, help='the number of epochs')

args = parser.parse_args()

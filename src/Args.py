import argparse

# DATASET = "movie"
# DATASET = "restaurant"
# DATASET = "book"
# DATASET = "music_all"
# DATASET = "music_10"
# DATASET = "music_30"

# DATASET = "music_len4"
# DATASET = "dianping-2"
# DATASET = "dianping-4"
# DATASET = "kkbox-2"
# DATASET = "kkbox-4"
# DATASET = "music_len4"
# DATASET = "music_len5"
# DATASET = "music_len6"
# DATASET = "music_len2"
# DATASET = "music"
# DATASET = "kkbox"
# DATASET = "ml-1m_rule_find"
# DATASET = "ml-1m"
# DATASET = "music_len4_50"
# DATASET = "music_len4_100"
# DATASET = "music_len4_200"
# DATASET = "music_len4_300"
# DATASET = "music_len4_random_all"
# DATASET = "music_len4_random_50"
# DATASET = "music_len4_random_100"
# DATASET = "music_len4_random_200"
# DATASET = "music_len4_random_300"

# DATASET = "music_len4_all_transe"
# DATASET = "music_len4_50_transe"
# DATASET = "music_len4_100_transe"
# DATASET = "music_len4_200_transe"
# DATASET = "music_len4_300_transe"

# DATASET ="lastfm_len5_300"
# DATASET ="lastfm_len5_150"
# DATASET ="lastfm_len5_100"
# DATASET ="lastfm_len5_200"
# DATASET ="lastfm_len5_250"

# DATASET ="music_len3_transe"
# DATASET ="music_len3_10_transe"
# DATASET ="music_len3_30_transe"
# DATASET ="music_len3_50_transe"

# DATASET ="music_len3_random"
# DATASET ="music_len3_10_random"
# DATASET ="music_len3_30_random"
# DATASET ="music_len3_50_random"

# DATASET = "kkbox_new""
parser = argparse.ArgumentParser()

# DATASET = "music_std_all"
# DATASET = "music_10"
# DATASET = "music_30"
# DATASET = "music_50"
# DATASET = "music_std_10"
# DATASET = "music_std_30"
# DATASET = "music_std_50"
# DATASET = "music_std_all"
# DATASET = "music_10_rlvlr"
# DATASET = "music_30_rlvlr"
# DATASET = "music_50_rlvlr"
# DATASET = "music_all_rlvlr"
DATASET = "music_correct_rotate_10"
# DATASET = "music_correct_rotate_30"
# DATASET = "music_correct_rotate_50"
# DATASET = "music_correct_rotate_all"
parser.add_argument('--rkgcn_dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--rkgcn_l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--rkgcn_lr', type=float, default=5e-2, help='learning rate')
parser.add_argument('--rkgcn_batch_size', type=int, default=128, help='batch size')
# parser.add_argument('--rkgcn_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--rkgcn_n_epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--rkgcn_dropout', type=float, default=0, help='probility to drop out')
parser.add_argument('--rkgcn_max_step', type=int, default=3, help='max length of rule')
parser.add_argument('--rkgcn_neighbour_size', type=int, default=4,
                    help="the num of neighbourhood to sample of an entity")

# DATASET = "restaurant"
# parser.add_argument('--rkgcn_dim', type=int, default=8, help='dimension of user and entity embeddings')
# parser.add_argument('--rkgcn_l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
# parser.add_argument('--rkgcn_lr', type=float, default=5e-4, help='learning rate')
# parser.add_argument('--rkgcn_batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--rkgcn_n_epochs', type=int, default=100, help='the number of epochs')
# parser.add_argument('--rkgcn_dropout', type=float, default=0, help='probility to drop out')
# parser.add_argument('--rkgcn_max_step', type=int, default=3, help='max length of rule')
# parser.add_argument('--rkgcn_neighbour_size', type=int, default=4,
#                     help="the num of neighbourhood to sample of an entity")


# # DATASET = "movie_1M"
# # DATASET = "movie_1M_10"
# DATASET = "movie_1M_30"
# # DATASET = "movie_1M_50"
# DATASET = "movie_1M_len4"

# parser.add_argument('--rkgcn_dim', type=int, default=8, help='dimension of user and entity embeddings')
# parser.add_argument('--rkgcn_l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
# parser.add_argument('--rkgcn_lr', type=float, default=5e-4, help='learning rate')
# parser.add_argument('--rkgcn_batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--rkgcn_n_epochs', type=int, default=100, help='the number of epochs')
# parser.add_argument('--rkgcn_dropout', type=float, default=0, help='probility to drop out')
# parser.add_argument('--rkgcn_max_step', type=int, default=3, help='max length of rule')
# parser.add_argument('--rkgcn_neighbour_size', type=int, default=4,
#                     help="the num of neighbourhood to sample of an entity")


'''
DATASET = "movie_20M"
parser.add_argument('--rkgcn_dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--rkgcn_l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--rkgcn_lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--rkgcn_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--rkgcn_n_epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--rkgcn_dropout', type=float, default=0, help='probility to drop out')
parser.add_argument('--rkgcn_max_step', type=int, default=3, help='max length of rule')
parser.add_argument('--rkgcn_neighbour_size', type=int, default=4,
                    help="the num of neighbourhood to sample of an entity")
'''

parser.add_argument('--rule_seg', type=str, default="->",
                    help='the signal to split r_name of a rule')
parser.add_argument('--kg_file', type=str, default="../data/" + DATASET + "/inv_kg_final",
                    help="file to store the whole kg.")
parser.add_argument("--user_nega_data_file", type=str, default="../data/" + DATASET + "/model/negative_data.txt",
                    help="file to store user record dict")
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

# parser.add_argument('--max_step', type=int, default=3, help="max length of rules")
# parser.add_argument('--max_step', type=int, default=4, help="max length of rules")


parser.add_argument('--sampled_ht_ratio', type=float, default=0.8,
                    help='the ratio of [h_id,t_id] of r_id used to search inference rules')
parser.add_argument("--chi_thresh", type=float, default=0.01, help="threshold for chi-square")
parser.add_argument('--filter_inv_pattern', type=bool, default=False,
                    help='remove rules with inverse and neighbouring relations')
parser.add_argument('--reserved_rule_num_by_frequency', type=int, default=10000,
                    help="the number of reserved rules by descending frequency")

# parser.add_argument("--aggregator_name", type=str, default="sum_aggregator", help="the type of aggregator")
parser.add_argument("--aggregator_name", type=str, default="concat_aggreator", help="the type of aggregator")
# parser.add_argument("--aggregator_name", type=str, default="neighbor_aggregator", help="the type of aggregator")


# rule gcn training params
parser.add_argument('--rkgcn_model_root_path', type=str, default='../data/' + DATASET + '/model/',
                    help="file to store rkgcn model")

# parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')


# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--ls_weight', type=float, default=0, help='weight of LS regularization')
# parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# pra training params
parser.add_argument('--pra_l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--pra_lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--pra_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--pra_n_epochs', type=int, default=100, help='the number of epochs')

parser.add_argument('--pra_model_file_path', type=str, default="../data/" + DATASET + "/model/pra_model.tar",
                    help="file to store the parameters of pra model")

parser.add_argument('--pre_train_rule_weight', type=bool, default=True, help="Using pre trained rule weight or not")
# parser.add_argument('--pre_train_rule_weight', type=bool, default=False, help="Using pre trained rule weight or not")
parser.add_argument('--freeze_rule_weight', type=bool, default=False, help="If freeze rule weight during training")
parser.add_argument('--rule_weight_file_path', type=str, default="../data/" + DATASET + '/model/rule_weight.npy',
                    help="file to store rule weight")

parser.add_argument('--train_feature_file', type=str, default="../data/" + DATASET + "/model/train_feature_label.npy",
                    help="file to store train feature data")
parser.add_argument('--eval_feature_file', type=str, default="../data/" + DATASET + "/model/eval_feature_label.npy",
                    help="file to store eval feature data.")
parser.add_argument('--test_feature_file', type=str, default="../data/" + DATASET + "/model/test_feature_label.npy",
                    help="file to store test feature data.")

args = parser.parse_args()


def args2str(my_args):
    res_str = ""
    for key, value in vars(my_args).items():
        res_str += "{}\t{}\n".format(key, value)
    return res_str


if __name__ == "__main__":
    print(list(vars(args)))
    print(args2str(args))

from Graph import Graph
from Args import args
import random
import numpy as np

# DATASET = "kkbox"
DATASET = "music_all"
# DATASET = "restaurant"
print(DATASET)
folder = "../data/" + DATASET + "/"

rule_file = folder + "model/" + "rule.txt"
g = Graph(args)
g.load_e_r_mapping()
g.construct_kg()

print("load rule")
rule_dict = {}
rule_token_str_dict = {}
with open(rule_file, "r", encoding="UTF-8") as f:
    for line in f.readlines():
        line = line.strip()

        rule_str = line.split("\t")[-1]
        rule_str = rule_str.strip()
        rule_dict[rule_str] = [g.r_id_by_name(rule_name) for rule_name in rule_str.split()]
        rule_token_str_dict[rule_str] = line

print("Get ground truth")
train_data = np.load(folder + "model/" + "train.npy")
eval_data = np.load(folder + "model/" + "eval.npy")
test_data = np.load(folder + "model/" + "test.npy")
ratings_np = np.concatenate([train_data, eval_data, test_data], axis=0)
ratings_np = ratings_np[np.where(ratings_np[:, -1] == 1)]
ratings_set = set()
for one_rating in ratings_np:
    user_id = one_rating[0]
    item_id = one_rating[1]
    ratings_set.add('{}_{}'.format(user_id, item_id))

print("Get Passed ht")
rule_line_conf_list = []
for rule_token in rule_dict:
    print(rule_token, end='\t')
    rule_id_list = rule_dict[rule_token]
    print(rule_id_list)
    passed_ht = g.get_passed_ht(rule_id_list)

    passed_ht_set = set()
    for ht in passed_ht:
        passed_ht_set.add("{}_{}".format(ht[0], ht[1]))

    correct_num = len(ratings_set & passed_ht_set)
    std_conf = correct_num / len(passed_ht_set)
    print("{}\t{}".format(rule_token, std_conf))
    rule_line_conf_list.append([rule_token_str_dict[rule_token], std_conf])

rule_line_conf_list = sorted(rule_line_conf_list, key=lambda x: x[1], reverse=True)

with open("{}std_conf_rule_sorted.txt".format(folder), 'w', encoding="UTF-8") as f:
    for one_rule_line in rule_line_conf_list:
        f.write("{}\n".format(one_rule_line[0]))
        print("{}\t{}".format(one_rule_line[1], one_rule_line[0]))

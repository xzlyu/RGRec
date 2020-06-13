from Graph import Graph
from Args import args
import random

DATASET = "kkbox"
# DATASET = "music"
# DATASET = "restaurant"
folder = "../data/" + DATASET + "/"

rule_file = folder + "model/" + "rule.txt"
g = Graph(args)
g.load_e_r_mapping()
g.construct_kg()

print("load rule")
rule_dict = {}
with open(rule_file, "r", encoding="UTF-8") as f:
    for line in f.readlines():
        rule_str = line.split("\t")[-1]
        rule_str = rule_str.strip()
        rule_dict[rule_str] = [g.r_id_by_name(rule_name) for rule_name in rule_str.split()]

print("Get Passed ht")

ht_writer = open(folder + "rule_ht.txt", 'w', encoding="UTF-8")
for rule_token in rule_dict:
    print(rule_token)
    ht_writer.write("{}\t".format(rule_token))
    rule_id_list = rule_dict[rule_token]
    print(rule_id_list)
    passed_ht = g.get_passed_ht(rule_id_list)


    if len(passed_ht) > 5000:
        passed_ht = random.sample(list(passed_ht), 5000)
    print("length of passed ht: {}".format(len(passed_ht)))

    ht_writer.write("{}\n".format(len(passed_ht)))

    for one_ht in passed_ht:
        ht_writer.write("{}\t{}\n".format(one_ht[0], one_ht[1]))
ht_writer.close()

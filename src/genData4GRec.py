import numpy as np
import os

DATASET = "movie_20M"
kg_final_file = "../data/{}/raw_data/kg_final.txt".format(DATASET)
ratings_final_file = "../data/{}/raw_data/ratings_final.txt".format(DATASET)

e_dict = {}
e_cnt = 0
r_dict = {}
r_cnt = 0


def add_new_entity(e_name):
    global e_cnt
    if e_name not in e_dict:
        e_dict[e_name] = e_cnt
        e_cnt += 1


def add_new_relation(r_name):
    global r_cnt
    if r_name not in r_dict:
        r_dict[r_name] = r_cnt
        r_cnt += 1


triple_list = []
with open(kg_final_file, 'r', encoding="UTF-8") as f:
    for line in f.readlines():
        h, r, t = line.strip().split()
        add_new_entity(h)
        add_new_entity(t)
        add_new_relation(r)

        inv_r = "inv_{}".format(r)
        add_new_relation(inv_r)

        triple_list.append([e_dict[h], r_dict[r], e_dict[t]])
        triple_list.append([e_dict[t], r_dict[inv_r], e_dict[h]])

add_new_relation("interact")
add_new_relation("inv_interact")

ratings_list = []
with open(ratings_final_file, 'r', encoding="UTF-8") as f:
    for line in f.readlines():
        user, item, label = line.strip().split()
        user = "u_{}".format(user)
        add_new_entity(user)
        if item not in e_dict:
            print("{} not in e_dict".format(item))
            continue
        label = int(label)
        ratings_list.append([e_dict[user], e_dict[item], label])

        triple_list.append([e_dict[user], r_dict["interact"], e_dict[item]])
        triple_list.append([e_dict[item], r_dict["inv_interact"], e_dict[user]])

result_folder = "../data/{}/".format(DATASET)
with open("{}entity_name2id.txt".format(result_folder), 'w', encoding="UTF-8") as f:
    f.write("{}".format(len(e_dict)))
    for e_name in e_dict:
        f.write("\n{}\t{}".format(e_name, e_dict[e_name]))

with open("{}relation_name2id.txt".format(result_folder), "w", encoding="UTF-8") as f:
    f.write("{}".format(len(r_dict)))
    for r_name in r_dict:
        f.write("\n{}\t{}".format(r_name, r_dict[r_name]))

with open("{}inv_kg_final.txt".format(result_folder), 'w', encoding="UTF-8") as f:
    for one_triple in triple_list:
        f.write("{}\t{}\t{}\n".format(one_triple[0], one_triple[1], one_triple[2]))

inv_kg_np = np.loadtxt("{}inv_kg_final.txt".format(result_folder), dtype=np.int64)
np.save("{}inv_kg_final.npy".format(result_folder), inv_kg_np)

with open("{}ratings_final.txt".format(result_folder), "w", encoding="UTF-8") as f:
    for one_rating in ratings_list:
        f.write("{}\t{}\t{}\n".format(one_rating[0], one_rating[1], one_rating[2]))

ratings_np = np.loadtxt("{}ratings_final.txt".format(result_folder), dtype=np.int64)
np.save("{}ratings_final.npy".format(result_folder), ratings_np)

# Split data
# train:eval:test =6:2:2
model_folder = "../data/{}/model/".format(DATASET)
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

rating_np_indices = np.array(range(len(ratings_np)))
np.random.shuffle(rating_np_indices)
train_indices, validate_indices, test_indices = np.split(rating_np_indices,
                                                         [int(.6 * len(rating_np_indices)),
                                                          int(.8 * len(rating_np_indices))])

np.save("{}train.npy".format(model_folder), ratings_np[train_indices, :])
np.save("{}eval.npy".format(model_folder), ratings_np[validate_indices, :])
np.save("{}test.npy".format(model_folder), ratings_np[test_indices, :])

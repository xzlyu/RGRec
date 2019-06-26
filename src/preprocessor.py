import numpy as np
import os

from src.Args import DATASET, args


def generate_relation_id_to_relation_name(DATASET):
    relation_id = 0
    relation_name2id = dict()
    kg_file_path = "../data/" + DATASET + "/kg.txt"
    with open(kg_file_path, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            array = line.split()
            relation_name = array[1]
            if relation_name not in relation_name2id:
                relation_name2id[relation_name] = relation_id
                relation_id += 1
    relation_name2id['interact'] = relation_id
    relation_id += 1

    total_relation_num = relation_id * 2

    relation_name2id_file_path = "../data/" + DATASET + "/relation_name2id.txt"
    inv_relation_name2id = dict()
    with open(relation_name2id_file_path, 'w', encoding="UTF-8") as f:

        f.write("Relation num {}\n".format(total_relation_num))
        for relation_name in relation_name2id:
            f.write("{}\t{}\n".format(relation_name, relation_name2id[relation_name]))
            inv_relation_name2id['inv_' + relation_name] = relation_id
            relation_id += 1
        for inv_relation_name in inv_relation_name2id:
            f.write("{}\t{}\n".format(inv_relation_name, inv_relation_name2id[inv_relation_name]))

    assert total_relation_num == relation_id, "Total: {}\tRelation id: {}".format(total_relation_num, relation_id)


# add user to kg
def construct_inv_kg(DATASET):
    relation_name2id = dict()
    relation_id2name = dict()
    entity_id_set = set()
    relation_name2id_file_path = "../data/" + DATASET + "/relation_name2id.txt"
    with open(relation_name2id_file_path, 'r', encoding="UTF-8") as f:
        for line in f.readlines()[1:]:
            relation_name, relation_id = line.split()
            relation_id = int(relation_id)
            relation_name2id[relation_name] = relation_id
            relation_id2name[relation_id] = relation_name

    writer = open('../data/' + DATASET + '/inv_kg_final.txt', 'w', encoding='utf-8')
    kg_file_path = "../data/" + DATASET + "/kg.txt"
    entity_id_max = -1
    for line in open(kg_file_path, 'r', encoding="UTF-8").readlines():
        head, relation_name, tail = line.split()
        entity_id_set.add(int(head))
        entity_id_set.add(int(tail))
        entity_id_max = max([int(head), int(tail), entity_id_max])
        relation_id = relation_name2id[relation_name]
        inv_relation_id = relation_name2id["inv_" + relation_name]
        writer.write("{}\t{}\t{}\n".format(head, relation_id, tail))
        writer.write("{}\t{}\t{}\n".format(tail, inv_relation_id, head))

    # add user to kg
    rating_file = '../data/' + DATASET + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    rating_np[:, 0] += entity_id_max
    converted_rating_file = "../data/" + DATASET + '/converted_ratings_final.npy'
    np.save(converted_rating_file, rating_np)

    interact_user_item = rating_np[np.where(rating_np[:, 2] == 1)]
    interact_id = relation_name2id["interact"]
    inv_interact_id = relation_name2id["inv_interact"]

    for array in interact_user_item:
        user_id = array[0]
        item_id = array[1]
        entity_id_set.add(user_id)
        assert item_id in entity_id_set, "Item id not in entity id set"
        writer.write("{}\t{}\t{}\n".format(user_id, interact_id, item_id))
        writer.write("{}\t{}\t{}\n".format(item_id, inv_interact_id, user_id))

    # entity_num = max(interact_user_item[:, 0])
    writer.close()

    with open("../data/" + DATASET + "/entity_name2id.txt", 'w', encoding="UTF-8") as f:
        f.write("Entity num {}".format(len(entity_id_set)))
        for e in entity_id_set:
            f.write("{}\t{}\n".format(e, e))


def split_train_eval_test(rating_file, train_file, eval_file, test_file):
    # train:eval:test 0.6:0.2:0.2

    rating_np = np.load(rating_file)

    rating_np_indices = np.array(range(len(rating_np)))
    np.random.shuffle(rating_np_indices)
    train_indices, validate_indices, test_indices = np.split(rating_np_indices,
                                                             [int(.6 * len(rating_np_indices)),
                                                              int(.8 * len(rating_np_indices))])

    np.save(train_file, rating_np[train_indices, :])
    np.save(eval_file, rating_np[validate_indices, :])
    np.save(test_file, rating_np[test_indices, :])

    # return list(train_data), list(eval_data), list(test_data)


if __name__ == "__main__":
    # generate_relation_id_to_relation_name(DATASET)
    # construct_inv_kg(DATASET)
    split_train_eval_test(args.converted_rating_file, args.train_file, args.eval_file, args.test_file)

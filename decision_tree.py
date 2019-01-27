import numpy as np
import matplotlib.pyplot as plt


def check_label(train_dataset):
    # Check all the label of the dataset
    # TODO:Check all the label of the train_dataset

    return False


def find_split(train_dataset):
    # TODO:Implement find_split
    return 0, 0


def split_dataset(train_dataset, attr, value):
    # Given attr:Int, value:Float, return two subsets of train_dataset
    left = [i for i in train_dataset if i["attrs"][attr] < value]
    right = [i for i in train_dataset if i["attrs"][attr] >= value]
    return left, right


def decision_tree_learning(train_dataset, depth):
    # train_dataset: [ {attrs:list, label:value}, ...]
    if check_label(train_dataset):
        # Base case: when all the label of dataset are the same then return a leaf node of the tree and the
        # corresponding depth
        node = {"attr": None, "value": None,
                "left": None, "right": None, "is_leaf": True, "label": train_dataset[0]["label"]}
        return node, depth
    else:
        # Find best split method
        split_attr, split_value = find_split(train_dataset)
        left_dataset, right_dataset = split_dataset(train_dataset, split_attr, split_value)
        left, l_depth = decision_tree_learning(left_dataset, depth + 1)
        right, r_depth = decision_tree_learning(right_dataset, depth + 1)
        node = {"attr": split_attr, "value": split_value,
                "left": left, "right": right, "is_leaf": False, "label": None}
        return node, max(l_depth, r_depth)


def predict(trained_node, data):
    # TODO
    # trained_node:: decision tree, data::{attrs:list, label:value}
    pass


def evaluate(trained_node, test_dataset, k=10):
    # TODO: 10 fold cross validation
    pass


file = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
train_dataset = [{"attrs": line[:-2], "label": line[-1]} for line in file]
print(train_dataset)

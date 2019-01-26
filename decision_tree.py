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


def construct_node(train_dataset, is_leaf=True, attr=None, value=None):
    # Split the dataset by the value of given attr
    left, right = split_dataset(train_dataset, attr, value)
    # Node structure: {attrs, left_subset, right_subset, label}, left and right subset are still the input dataset
    # format
    if is_leaf:
        return {"attrs": train_dataset[0]['attrs'], "left": train_dataset[1:],
                "right": [], "label": train_dataset[0]['label']}
    else:
        if attr is None:
            raise Exception("Attribute not define")
        if value is None:
            raise Exception("Value not define")
        return {"attrs": train_dataset[0]['attrs'], "left": left, "right": right, "label": train_dataset[0]['label']}


def decision_tree_learning(train_dataset, depth):
    # train_dataset: [ {attrs:list, label:value}, ...]
    if check_label(train_dataset):
        # Base case: when all the label of dataset are the same then return a left node of the tree and the
        # corresponding label
        node = construct_node(train_dataset)
        return node, train_dataset[0]['label']
    else:
        # TODO: Finish the recursive structure
        # Find best split method
        split_attr, split_value = find_split(train_dataset)


file = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
train_dataset = [{"attrs": line[:-2], "label":line[-1]} for line in file]
print(train_dataset)

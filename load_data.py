import numpy as np
import matplotlib.pyplot as plt

list = []
file = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
print(file[0])
# attributes = [f[3] for f in file]
# plt.plot(attributes)
# plt.show()

def check_label(train_dataset):
    # Check all the label of the dataset
    pass


def find_split(train_dataset):
    pass


def split_dataset(train_dataset, attr, value):
    # Given attr:Int, value:Float, return two subsets of train_dataset
    left = [i for i in train_dataset if i["attrs"][attr] < value]
    right = [i for i in train_dataset if i["attrs"][attr] >= value]
    return left, right


def construct_node(train_dataset, is_leaf, attr, value):
    # Split the dataset by the value of given attr
    left, right = split_dataset(train_dataset, attr, value)
    if is_leaf:
        return {"attrs":train_dataset[0]['attrs'], "left":train_dataset[1:],
                "right":[], "label":train_dataset[0]['label']}
    else:
        return {"attrs":train_dataset[0]['attrs'], "left":left, "right":right, "label":train_dataset[0]['label']}


def decision_tree_learning(train_dataset, depth):
    # train_dataset: [ {attrs:list, label:value}, ...]
    if check_label(train_dataset):
        node = construct_node(train_dataset)
        return node, train_dataset[0]['label']
    else:
        # Find best split method
        split = find_split(train_dataset)



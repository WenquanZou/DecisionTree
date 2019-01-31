import numpy as np
import matplotlib.pyplot as plt


def check_label(train_dataset):
    # Check all the label of the dataset
    current_label = train_dataset[0]["label"]
    for data_dict in train_dataset:
        if current_label != data_dict["label"]:
            return False
    return True


def find_entropy_attr(dataset, target_attr):
    unique_value = set()
    value_set = [data["attrs"][target_attr] for data in dataset]
    entropy = 0
    for data in dataset:
        unique_value.add(data["attrs"][target_attr])
    for v in unique_value:
        count = value_set.count(v)
        prob = count / len(value_set)
        entropy = entropy + -prob * np.log2(prob)
    return entropy


def find_entropy_value(entropy_total, dataset, target):
    unique_value = set()
    best_split_value = 0
    max_gain = 0
    for data in dataset:
        unique_value.add(data["attrs"][target])
    for v in unique_value:
        greater_dataset = list(filter(lambda x: x["attrs"][target] >= v, dataset))
        smaller_dataset = list(filter(lambda x: x["attrs"][target] < v, dataset))
        g_entropy = find_entropy_attr(greater_dataset, target)
        s_entropy = find_entropy_attr(smaller_dataset, target)
        remainder = len(greater_dataset) / len(dataset) * g_entropy + len(smaller_dataset) / len(dataset) * s_entropy
        gain = entropy_total - remainder
        if gain > max_gain:
            max_gain = gain
            best_split_value = v
    return best_split_value


def find_split(train_dataset):
    split_attr = None
    max_entropy = 0
    for i in range(len(train_dataset[0]["attrs"])):
        cur_entropy = find_entropy_attr(train_dataset, i)
        if max_entropy < cur_entropy:
            max_entropy = cur_entropy
            split_attr = i
    return split_attr, find_entropy_value(max_entropy, train_dataset, split_attr)


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
        # Split the training set
        left_dataset, right_dataset = split_dataset(train_dataset, split_attr, split_value)
        # Recursive call
        left, l_depth = decision_tree_learning(left_dataset, depth + 1)
        right, r_depth = decision_tree_learning(right_dataset, depth + 1)
        # Construct root node
        node = {"attr": split_attr, "value": split_value,
                "left": left, "right": right, "is_leaf": False, "label": None}
        return node, max(l_depth, r_depth)


def predict(trained_node, data):
    # trained_node:: decision tree, data::list
    if trained_node["is_leaf"]:
        return trained_node["label"]
    elif data[trained_node["attr"]] < trained_node["value"]:
        left = trained_node["left"]
        return predict(left, data)
    else:
        right = trained_node["right"]
        return predict(right, data)

def evaluate(test_dataset, trained_tree):
    # TODO: 10 fold cross validation + Metric(Eric)
    pass


file = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
train_dataset = [{"attrs": list(line[:-1]), "label": line[-1]} for line in file]
node, _ = decision_tree_learning(train_dataset, 0)
print(predict(node, [-52,-55,-52,-43,-61,-86,-83]))

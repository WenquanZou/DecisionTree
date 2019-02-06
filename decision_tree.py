import numpy as np


def decision_tree_learning(train_dataset, depth):
    # train_dataset: [ {attrs:list, label:value}, ...]
    if check_label(train_dataset):
        # Base case: when all the label of dataset are the same then return a leaf node of the tree and the
        # corresponding depth
        return {"attr": None, "value": None,
                "left": None, "right": None, "is_leaf": True, "label": train_dataset[0]["label"]}, depth
    else:
        # Find best split method
        split_attr, split_value = find_split(train_dataset)
        # Split the training set
        left_dataset, right_dataset = split_dataset(train_dataset, split_attr, split_value)
        # Recursive call
        left, l_depth = decision_tree_learning(left_dataset, depth + 1)
        right, r_depth = decision_tree_learning(right_dataset, depth + 1)
        # Construct root node
        return {"attr": split_attr, "value": split_value,
                "left": left, "right": right, "is_leaf": False, "label": None}, max(l_depth, r_depth)


def check_label(train_dataset):
    # Check all the label of the dataset
    current_label = train_dataset[0]["label"]
    for data_dict in train_dataset:
        if current_label != data_dict["label"]:
            return False
    return True


def find_entropy(dataset):
    value_set = [data["label"] for data in dataset]
    unique_value = set(value_set)
    entropy = 0
    for v in unique_value:
        # Every distinct value with its probability discretely
        count = value_set.count(v)
        prob = count / len(value_set)
        # Compute entropy for the value
        entropy = entropy + -prob * np.log2(prob)
    return entropy


def find_entropy_attr(dataset, target_attr):
    value_set = [data["attrs"][target_attr] for data in dataset]
    unique_value = set(value_set)
    entropy = find_entropy(dataset)
    for v in unique_value:
        # Every distinct value with its probability discretely
        count = value_set.count(v)
        proportion = count / len(value_set)
        filtered_dataset = [data for data in dataset if data["attrs"][target_attr] == v]
        # Compute entropy for the value
        entropy = entropy - proportion * find_entropy(filtered_dataset)
    return entropy


def find_information_gain(entropy_total, dataset, target):
    # Find the entropy of an attribute in every possible value to split and return the best split value
    #  from this attribute
    unique_value = set()
    best_split_value = 0
    max_gain = 0
    for data in dataset:
        # Distinct values
        unique_value.add(data["attrs"][target])
    for v in unique_value:
        # Generate subset for greater and smaller
        greater_dataset = list(filter(lambda x: x["attrs"][target] >= v, dataset))
        smaller_dataset = list(filter(lambda x: x["attrs"][target] < v, dataset))
        # Compute each entropy for both subset
        g_entropy = find_entropy_attr(greater_dataset, target)
        s_entropy = find_entropy_attr(smaller_dataset, target)
        remainder = len(greater_dataset) / len(dataset) * g_entropy + len(smaller_dataset) / len(dataset) * s_entropy
        gain = entropy_total - remainder
        if gain > max_gain:
            max_gain = gain
            # Reassign split value if greater gain is found
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
    return split_attr, find_information_gain(max_entropy, train_dataset, split_attr)


def split_dataset(train_dataset, attr, value):
    # Given attr:Int, value:Float, return two subsets of train_dataset
    left = [i for i in train_dataset if i["attrs"][attr] < value]
    right = [i for i in train_dataset if i["attrs"][attr] >= value]
    return left, right


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


def cross_fold_split(train_dataset):
    # Split up the dataset and double shuffle
    dataset = list()
    train_dataset_copy = list(train_dataset)
    fold_size = int(len(train_dataset) / 10)
    for i in range(10):
        fold_set = list()
        while len(fold_set) < fold_size:
            rand_index = np.random.randint(0, len(train_dataset_copy))
            fold_set.append(train_dataset_copy.pop(rand_index))
        dataset.append(fold_set)
    return dataset


def c_matrix(actual, predicted, n):
    # Generate confusion matrix
    cmatrix = np.zeros((n, n))
    for i in range(len(actual)):
        cmatrix[int(actual[i]) - 1][int(predicted[i]) - 1] += 1
    return cmatrix


def cross_validation(dataset):
    # Split dataset into 10 folds
    folds = cross_fold_split(dataset)
    max_accuracy = -1
    best_tree = None

    precision_matrix = []
    recall_matrix = []
    f1_data_matrix = []
    accuracy_list = []

    for fold in folds:
        # Choose every fold as test set in random order
        train_folds = list(folds)
        train_folds.remove(fold)
        train_folds = sum(train_folds, [])
        test_fold = list(fold)
        # Training process
        dtree, depth = decision_tree_learning(train_folds, 0)
        print(f'Depth is {depth}')
        # Evaluate for every class
        precision_list, recall_list, f1_data_list, accuracy = evaluate(dtree, test_fold)
        precision_matrix.append(precision_list)
        recall_matrix.append(recall_list)
        f1_data_matrix.append(f1_data_list)
        accuracy_list.append(accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_tree = dtree

    # Average result for every class of label
    average_precision = [np.mean(x) for x in zip(*precision_matrix)]
    average_recall = [np.mean(x) for x in zip(*recall_matrix)]
    average_f1_data = [np.mean(x) for x in zip(*f1_data_matrix)]
    # Average accuracy of ten iterations of accuracy
    average_accuracy = np.mean(accuracy_list)

    print(f'precision for every class: {average_precision}\n'
          f'recall for every class: {average_recall}\n'
          f'f1_data for every class: {average_f1_data}\n'
          f'accuracy: {average_accuracy}')
    return best_tree


def evaluate(dtree, test_fold):
    predicted_labels = list()
    actual_labels = list()
    # Get metrics for every testing fold
    for data in test_fold:
        predicted_labels.append(predict(dtree, data["attrs"]))
        actual_labels.append(data["label"])

    # Generate confusion matrix
    confusion_matrix = c_matrix(actual_labels, predicted_labels, 4)
    # Keep track on precision and recall of every label to sum
    precision_list = []
    recall_list = []
    accuracy_sum = 0
    total = 0
    for i in range(4):
        # One label precision and recall
        precision_denom = 0
        recall_denom = 0
        for j in range(4):
            precision_denom += confusion_matrix[i][j]
            recall_denom += confusion_matrix[j][i]
            total += confusion_matrix[i][j]
        precision_list.append(confusion_matrix[i][i] / precision_denom)
        recall_list.append(confusion_matrix[i][i] / recall_denom)
        accuracy_sum += confusion_matrix[i][i]

    f1_data_list = list(map(lambda p, r: 2 * p * r / (p + r), precision_list, recall_list))
    accuracy = accuracy_sum / total
    return precision_list, recall_list, f1_data_list, accuracy


def prune(root, dataset, node=None):
    if node is None:
        node = root

    if node['is_leaf']:
        return node

    node['left'] = prune(root, dataset, node['left'])
    node['right'] = prune(root, dataset, node['right'])

    left = node['left']
    right = node['right']

    if left['is_leaf'] and right['is_leaf']:
        _, _, _, original_score = evaluate(root, dataset)

        node['is_leaf'] = True

        node['label'] = left['label']
        _, _, _, left_score = evaluate(root, dataset)

        node['label'] = right['label']
        _, _, _, right_score = evaluate(root, dataset)

        changed_to_leaf = False

        if left_score > original_score:
            original_score = left_score
            node['label'] = left['label']
            changed_to_leaf = True

        if right_score > original_score:
            node['label'] = right['label']
            changed_to_leaf = True

        if not changed_to_leaf:
            node['is_leaf'] = False
            node['label'] = None

    return node


clean = 'co395-cbc-dt/wifi_db/clean_dataset.txt'
noisy = 'co395-cbc-dt/wifi_db/noisy_dataset.txt'

file = np.loadtxt(noisy)
train_dataset = [{"attrs": list(line[:-1]), "label": line[-1]} for line in file]
node, _ = decision_tree_learning(train_dataset, 0)
best_tree = cross_validation(train_dataset)

clean_dataset = train_dataset
noisy_dataset = [{"attrs": list(line[:-1]), "label": line[-1]} for line in
                 np.loadtxt('co395-cbc-dt/wifi_db/noisy_dataset.txt')]

_, _, _, before_clean_metrics = evaluate(best_tree, clean_dataset)
_, _, _, before_noisy_metrics = evaluate(best_tree, noisy_dataset)
pruned_tree = prune(best_tree, noisy_dataset)
_, _, _, after_clean_metrics = evaluate(pruned_tree, clean_dataset)
_, _, _, after_noisy_metrics = evaluate(pruned_tree, noisy_dataset)

print(f'Clean dataset (before): {before_clean_metrics}')
print(f'Noisy dataset (before): {before_noisy_metrics}')
print(f'Clean dataset (after): {after_clean_metrics}')
print(f'Noisy dataset (after): {after_noisy_metrics}')

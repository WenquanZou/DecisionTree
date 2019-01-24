import numpy as np
import matplotlib.pyplot as plt

list = []
file = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
print(file[0])
# attributes = [f[3] for f in file]
# plt.plot(attributes)
# plt.show()

def check_label(train_dataset):
    pass


def find_split(train_dataset):
    pass


def decision_tree_learning(train_dataset, depth):
    if check_label(train_dataset):
        return train_dataset[0][-1]
    else:
        # Find best split method
        split = find_split(train_dataset)



import numpy as np
from CNN import CNN
from costFunctions import CostOLS
from sklearn.datasets import load_digits

def onehot(target: np.ndarray):
    onehot = np.zeros((target.size, target.max() + 1))
    onehot[np.arange(target.size), target] = 1
    return onehot

digits = load_digits()

X = digits.data
target = digits.target
target = onehot(target)

cost_func = CostOLS(target)
cnn = CNN(cost_func)

from src.Layers import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from src.FFNN import FFNN
from src.CNN import CNN
import numpy as np
np.random.seed(2023)

"""
Test file too see if FullyConnectedLayer functions as feed forward neural network (FFNN)
"""

# simple dataset
cancer_dataset = load_breast_cancer()
cancer_X = cancer_dataset.data
cancer_t = cancer_dataset.target
cancer_t = cancer_t.reshape(cancer_t.shape[0], 1)

X_train, X_val, t_train, t_val = train_test_split(cancer_X, cancer_t)
scaler = MinMaxScaler()
scaler.fit(X_train) 
X_train = scaler.transform(X_train) 
X_val = scaler.transform(X_val) 

rho = 0.9
rho2 = 0.999
eta = 1e-4
lam = 1e-4
momentum = 0.9

seed = 2023 

adam_scheduler= Adam(eta, rho, rho2)
momentum_scheduler = Momentum(eta, momentum)
dims1 = [cancer_X.shape[1], 50]

layer1 = FullyConnectedLayer(
        dims1,
        sigmoid, 
        Adam, 
        seed=seed
        )

layer_a = layer1._feedforward(X_train)
# print(f"{layer1.weights=}")
# print(f"{layer_a=}")

cnn = CNN(seed=seed)

# test way to connect layers
cnn.FullyConnectedLayer(
        dims1,
        sigmoid, 
        Adam, 
        seed=seed
        )

cnn.FullyConnectedLayer(
        60,
        sigmoid, 
        Adam, 
        seed=seed
        )

cnn.FullyConnectedLayer(
        70,
        sigmoid, 
        Adam, 
        seed=seed
        )

cnn_a = cnn._feedforward(X_train)
print(f"{cnn_a=}")

ffnn = FFNN(dims1, sigmoid, seed=seed)
ffnn_a = ffnn._feedforward(X_train)

# TODO this should work, unsure what problem is
assert (layer_a==ffnn_a).all(), "feedforward output not equal in FFNN and FullyConnectedLayer"

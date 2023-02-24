from src.Layers import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from src.FFNN import FFNN
from src.CNN import CNN
from src.costFunctions import CostOLS
import autograd.numpy as np


"""
Test file to test differnet layers and combinations of layers in a CNN 
"""

# simple dataset
cancer_dataset = load_breast_cancer()
cancer_X = cancer_dataset.data
cancer_t = cancer_dataset.target
cancer_t = cancer_t.reshape(cancer_t.shape[0], 1)

np.random.seed(1337)
X_train, X_val, t_train, t_val = train_test_split(cancer_X, cancer_t)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)


seed = 1337
rho = 0.9
rho2 = 0.999
momentum = 0.9
eta = 1e-3
lam = 1e-5
epochs = 200

batches = 10
feature_maps = 10
print(f"{X_train.shape=}")

H = X_train.shape[0]
W = X_train.shape[1] // feature_maps

print(W)

reshaped_X_train = np.zeros((batches, feature_maps, H, W))
index_counter = 0
for b in range(batches):
    for fm in range(feature_maps):
        for h in range(H):
            for w in range(W):
                reshaped_X_train[b, fm, h, w] = X_train[h, w + fm * W]

print(f"{reshaped_X_train.shape=}")
test_flatten = FlattenLayer(seed)
flattened_reshaped = test_flatten._feedforward(reshaped_X_train)
print(f"{flattened_reshaped.shape=}")


adam_scheduler = Adam(eta, rho, rho2)
momentum_scheduler = Momentum(eta, momentum)

cnn = CNN(scheduler=adam_scheduler, seed=seed)

# test way to connect layers
print(f"{seed=}")
#
# cnn.add_FlattenLayer(seed=seed)
#
# cnn.add_FullyConnectedLayer(X_train.shape[1], LRELU, seed=seed)
#
# cnn.add_FullyConnectedLayer(100, LRELU, seed=seed)
#
# cnn.add_OutputLayer(1, sigmoid, seed=seed)
#
# cnn._feedforward(reshaped_X_train)
# # cnn.fit(
# #     reshaped_X_train, t_train, lam=lam, batches=batches, epochs=epochs, X_val=X_val, t_val=t_val
# # )
#

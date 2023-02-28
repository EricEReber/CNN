from src.Layers import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from src.FFNN import FFNN
from src.CNN import CNN
from src.costFunctions import CostOLS
import autograd.numpy as np
from src.Layers import FlattenLayer
import matplotlib.pyplot as plt
import imageio.v3 as imageio

"""
Test file to test replacing progress bar with ProgressBar class
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

print(f"{X_train.shape=}")
print(f"{X_val.shape=}")

seed = 1337
rho = 0.9
rho2 = 0.999
momentum = 0.9
eta = 1e-3
lam = 1e-5
epochs = 200
batches = X_train.shape[0]
feature_maps = 1

H = X_train.shape[1] // 2
W = X_train.shape[1] // 2 // feature_maps

reshaped_X_train = np.zeros((batches, feature_maps, H, W))
for b in range(batches):
    for fm in range(feature_maps):
        for h in range(H):
            for w in range(W):
                reshaped_X_train[b, fm, h, w] = X_train[h, w + fm * W]
print(f"{reshaped_X_train.shape=}")

batches = X_val.shape[0]
H = X_val.shape[1] // 2
W = X_val.shape[1] // 2 // feature_maps
reshaped_X_val = np.zeros((batches, feature_maps, H, W))
for b in range(batches):
    for fm in range(feature_maps):
        for h in range(H):
            for w in range(W):
                reshaped_X_val[b, fm, h, w] = X_val[h, w + fm * W]
print(f"{reshaped_X_val.shape=}")

adam_scheduler = Adam(eta, rho, rho2)

cnn = CNN(scheduler=adam_scheduler, seed=seed)

cnn.add_FlattenLayer(seed=seed)

cnn.add_FullyConnectedLayer(feature_maps * H * W, LRELU, seed=seed)

cnn.add_FullyConnectedLayer(100, LRELU, seed=seed)

cnn.add_OutputLayer(1, sigmoid, seed=seed)

cnn.fit(
    reshaped_X_train,
    t_train,
    lam=lam,
    batches=batches,
    epochs=epochs,
    X_val=reshaped_X_val,
    t_val=t_val,
)

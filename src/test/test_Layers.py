from src.Layers import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from src.FFNN import FFNN
from src.CNN import CNN
from src.costFunctions import CostOLS
import autograd.numpy as np
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
X_train = np.array([[5, 2, 3]])
target = np.array([[1, 2, 3]])
dims1 = [X_train.shape[1], 2]

cnn = CNN(seed=seed)

# test way to connect layers
cnn.add_FullyConnectedLayer(
        dims1,
        identity, 
        seed=seed
        )

cnn.add_FullyConnectedLayer(
        5,
        identity, 
        seed=seed
        )

cnn.add_OutputLayer(3, identity, seed=seed)

out_test = OutputLayer([1,1], identity, CostOLS, Adam)
print(isinstance(out_test, FullyConnectedLayer))
cnn_a = cnn.fit(X_train, target)
print(f"{cnn_a=}")
# TODO for some reason outputlayer makes FullyConnectedLayer work worse?
# unsure, ill figure it out


ffnn = FFNN(dims1, sigmoid, seed=seed)
ffnn_a = ffnn._feedforward(X_train)

# TODO this should work, unsure what problem is
# assert (layer_a==ffnn_a).all(), "feedforward output not equal in FFNN and FullyConnectedLayer"

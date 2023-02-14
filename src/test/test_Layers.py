from src.Layers import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import src.FFNN
import numpy as np


cancer_dataset = load_breast_cancer()
cancer_X = cancer_dataset.data
cancer_t = cancer_dataset.target
cancer_t = cancer_t.reshape(cancer_t.shape[0], 1)


rho = 0.9
rho2 = 0.999
adam_eta = 1e-4
adam_lambda = 1e-4


np.random.seed(1337)
X_train, X_val, t_train, t_val = train_test_split(cancer_X, cancer_t)
scaler = MinMaxScaler()
scaler.fit(X_train) 
X_train = scaler.transform(X_train) 
X_val = scaler.transform(X_val) 

adam_params = [adam_eta, rho, rho2]
momentum_params = [0.02, 0.9]

layer = FullyConnectedLayer(
        [cancer_X.shape[1], 50],
        sigmoid, 
        Adam, 
        *adam_params,
        )

print(layer.weights)

layer._feedforward(X_train)

print(layer.a_matrix)

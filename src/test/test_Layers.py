import sys 
sys.path.append('/home/gregz/Files/CNN/src/')
from Layers import *
import FFNN
from Schedulers import *
from activationFunctions import *
from costFunctions import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np

# cancer_dataset = load_breast_cancer()
# cancer_X = cancer_dataset.data
# cancer_t = cancer_dataset.target
# cancer_t = cancer_t.reshape(cancer_t.shape[0], 1)


rho = 0.9
rho2 = 0.999
adam_eta = 1e-4
adam_lambda = 1e-4


adam_params = [adam_eta, rho, rho2]
momentum_params = [0.02, 0.9]

layer = FullyConnectedLayer(
        [100, 50],
        sigmoid, 
        Adam, 
        *adam_params,
        )

print(layer.weights)

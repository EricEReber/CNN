from src.Layers import *
from sklearn.datasets import load_digits
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
Test file to test differnet layers and combinations of layers in a CNN 
"""

# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_val, y_val) = mnist.load_data()

dataset = load_digits()
mnist = dataset["images"]
target = dataset["target"]

x_train, x_val, y_train, y_val = train_test_split(mnist, target)
print(f"{x_train.shape=}")
print(f"{x_val.shape=}")
print(f"{y_train.shape=}")

x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)

scaler = MinMaxScaler()
scaler.fit(x_train[0, 0, :, :])
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
        x_train[i,j, :, :]= scaler.transform(x_train[i, j, :, :])
        # x_val = scaler.transform(x_val)

for i in range(x_val.shape[0]):
    for j in range(x_val.shape[1]):
        x_val[i,j, :, :]= scaler.transform(x_val[i, j, :, :])
# simple dataset
np.random.seed(1337)

seed = 1337
rho = 0.9
rho2 = 0.999
momentum = 0.9
eta = 1e-3
lam = 1e-5
epochs = 200

images = 12
batches = 6
feature_maps = 11

adam_scheduler = Adam(eta, rho, rho2)
momentum_scheduler = Momentum(eta, momentum)

cnn = CNN(scheduler=adam_scheduler, seed=seed)

# test way to connect layers
#
# cnn.add_Convolution2DLayer(kernel_height=2, kernel_width=2, optimized=False)
cnn.add_FlattenLayer()

cnn.add_FullyConnectedLayer(100, LRELU)

cnn.add_FullyConnectedLayer(100, LRELU)

cnn.add_OutputLayer(10, sigmoid)

cnn.fit(
    x_train,
    y_train,
    lam=lam,
    batches=batches,
    epochs=epochs,
    X_val=x_val,
    t_val=y_val,
)

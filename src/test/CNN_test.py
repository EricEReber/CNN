from src.Layers import *
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.FFNN import FFNN
from src.CNN import CNN
from src.costFunctions import *
import autograd.numpy as np
from src.Layers import FlattenLayer
import matplotlib.pyplot as plt
import imageio.v3 as imageio

"""
Test file to test differnet layers and combinations of layers in a CNN 
"""


def onehot(target: np.ndarray):
    onehot = np.zeros((target.size, target.max() + 1))
    onehot[np.arange(target.size), target] = 1
    return onehot


# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_val, y_val) = mnist.load_data()

dataset = load_digits()
mnist = dataset["images"]
target = dataset["target"]
target = onehot(target)

x_train, x_val, y_train, y_val = train_test_split(mnist, target)

x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
x_train = x_train[:, :, :x_val.shape[2]-4, :]
x_val = x_val[:, :, :x_val.shape[2]-4, :]
# y_train = y_train.reshape(y_train.shape[0], 1)
# y_val = y_val.reshape(y_val.shape[0], 1)
print(f"{x_train.shape=}")
print(f"{x_val.shape=}")
print(f"{y_train.shape=}")

scaler = MinMaxScaler()
scaler.fit(x_train[0, 0, :, :])
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
        x_train[i, j, :, :] = scaler.transform(x_train[i, j, :, :])
        # x_val = scaler.transform(x_val)

for i in range(x_val.shape[0]):
    for j in range(x_val.shape[1]):
        x_val[i, j, :, :] = scaler.transform(x_val[i, j, :, :])

# simple dataset
np.random.seed(1337)

seed = 1337
rho = 0.9
rho2 = 0.999
momentum = 0.9
eta = 1e-3
lam = 1e-5
epochs = 50

images = 12
batches = 6

adam_scheduler = Adam(eta, rho, rho2)
momentum_scheduler = Momentum(eta, momentum)

cnn = CNN(cost_func=CostCrossEntropy, scheduler=adam_scheduler, seed=seed)

# FOR odd-whole kernel bug
# cnn.add_Convolution2DLayer(
#     input_channels=1, feature_maps=1, kernel_height=2, kernel_width=6, optimized=True
# )
# cnn.add_Convolution2DLayer(
#     input_channels=1, feature_maps=1, kernel_height=2, kernel_width=3, optimized=True
# )


cnn.add_Convolution2DLayer(
    act_func=LRELU,
    input_channels=1,
    feature_maps=1,
    kernel_height=2,
    kernel_width=6,
    optimized=True,
)
cnn.add_PoolingLayer(kernel_width=3, kernel_height=3, v_stride=1, h_stride=1, pooling="max")

cnn.add_Convolution2DLayer(
    act_func=LRELU,
    input_channels=1,
    feature_maps=1,
    kernel_height=5,
    kernel_width=3,
    optimized=False,
)
cnn.add_FlattenLayer()

cnn.add_FullyConnectedLayer(30, LRELU)

cnn.add_FullyConnectedLayer(20, LRELU)

cnn.add_OutputLayer(10, softmax)

cnn.fit(
    x_train,
    y_train,
    lam=lam,
    batches=batches,
    epochs=epochs,
    X_val=x_val,
    t_val=y_val,
)

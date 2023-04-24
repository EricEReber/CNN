from src.Layers import *
from sklearn.datasets import fetch_openml
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
Test file to test capabilites of CNN.py on 28x28 MNIST data
"""


def onehot(target: np.ndarray):
    onehot = np.zeros((target.size, target.max() + 1))
    onehot[np.arange(target.size), target] = 1
    return onehot


dataset = fetch_openml("mnist_784", parser="auto")
mnist = dataset.data.to_numpy(dtype="float")[:10000, :]
print(f"{mnist.shape=}")
print(f"{mnist.shape=}")
for i in range(mnist.shape[1]):
    mnist[:, i] /= 255
mnist = mnist.reshape(mnist.shape[0], 1, 28, 28)
target = onehot(np.array([int(i) for i in dataset.target.to_numpy()[:10000]]))

x_train, x_val, y_train, y_val = train_test_split(mnist, target)
seed = 1337
np.random.seed(seed)
rho = 0.9
rho2 = 0.999
momentum = 0.9
eta = 1e-3
lam = 1e-5
epochs = 200
batches =11

adam_scheduler = Adam(eta, rho, rho2)
momentum_scheduler = Momentum(eta, momentum)

cnn = CNN(cost_func=CostCrossEntropy, scheduler=adam_scheduler, seed=seed)

cnn.add_Convolution2DLayer(
    act_func=LRELU,
    input_channels=1,
    feature_maps=1,
    kernel_height=2,
    kernel_width=2,
    v_stride=1,
    h_stride=1,
    optimized=True,
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

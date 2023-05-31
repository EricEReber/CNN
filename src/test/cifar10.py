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
from src.utils import accuracy, confusion, plot_confusion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

"""
Test file to test capabilites of CNN.py on 32x32 CIFAR-10 data
"""
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Set a random seed for reproducibility
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(conf_matrix, classes):
    normalized_matrix = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)
    fig, ax = plt.subplots()
    ax = sns.heatmap(normalized_matrix, annot=True, cmap="Blues", fmt=".2%", cbar=False)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.title("Confusion Matrix")
    plt.show()


def onehot(target: np.ndarray):
    onehot = np.zeros((target.size, target.max() + 1))
    onehot[np.arange(target.size), target] = 1
    return onehot


seed = 1337
np.random.seed(seed)

# Download the CIFAR-10 dataset and apply transformations
transform = transforms.Compose([transforms.ToTensor()])

cifar10_train = CIFAR10(root="./data", train=True, download=True, transform=transform)
cifar10_test = CIFAR10(root="./data", train=False, download=True, transform=transform)

# Get the features and labels from the CIFAR-10 dataset
X = cifar10_train.data
X = X.reshape(X.shape[0], X.shape[3], X.shape[1], X.shape[2])
y = np.array(cifar10_train.targets)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
X_scaled = X_scaled.reshape(X.shape)

# Split the dataset into training and testing sets
x_train, x_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y
)

labels = np.arange(0, 10, 1)
rho = 0.9
rho2 = 0.999
momentum = 0.9
eta = 1e-3
lam = 1e-5
epochs = 100
batches = 10

adam_scheduler = Adam(eta, rho, rho2)
momentum_scheduler = Momentum(eta, momentum)

cnn = CNN(cost_func=CostCrossEntropy, scheduler=adam_scheduler, seed=seed)

cnn.add_Convolution2DLayer(
    act_func=LRELU,
    input_channels=3,
    feature_maps=3,
    kernel_height=2,
    kernel_width=2,
    v_stride=3,
    h_stride=3,
    optimized=True,
)
# cnn.add_Convolution2DLayer(
#     act_func=LRELU,
#     input_channels=3,
#     feature_maps=6,
#     kernel_height=3,
#     kernel_width=3,
#     v_stride=1,
#     h_stride=1,
#     optimized=True,
# )
# cnn.add_PoolingLayer(kernel_height=2, kernel_width=2, pooling="max")

cnn.add_FlattenLayer()

cnn.add_FullyConnectedLayer(30, LRELU)

cnn.add_FullyConnectedLayer(20, LRELU)

cnn.add_OutputLayer(10, softmax)

scores = cnn.fit(
    x_train[::4],
    y_train[::4],
    lam=lam,
    batches=batches,
    epochs=epochs,
    X_val=x_val[::4],
    t_val=y_val[::4],
)

plt.plot(scores["train_acc"], label="Training")
plt.plot(scores["val_acc"], label="Validation")
plt.ylim([0.8, 1])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


prediction = cnn.predict(x_val[::4])
conf_array = plot_confusion_matrix(
    np.argmax(prediction, axis=-1), np.argmax(y_val[::4], axis=-1)

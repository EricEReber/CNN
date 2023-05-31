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

"""
Test file to test capabilites of CNN.py on 28x28 MNIST data
"""
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


dataset = fetch_openml("mnist_784", parser="auto")
mnist = dataset.data.to_numpy(dtype="float")[:10000, :]
# mnist_test = dataset.data.to_numpy(dtype="float")[10000::50, :]
for i in range(mnist.shape[1]):
    mnist[:, i] /= 255
mnist = mnist.reshape(mnist.shape[0], 1, 28, 28)
# mnist_test = mnist_test.reshape(mnist_test.shape[0], 1, 28, 28)
target = onehot(np.array([int(i) for i in dataset.target.to_numpy()[:10000]]))
# target_test = onehot(np.array([int(i) for i in dataset.target.to_numpy()[10000::50]]))

x_train, x_val, y_train, y_val = train_test_split(mnist, target)
labels = np.arange(0, 10, 1)
seed = 1337
np.random.seed(seed)
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
    input_channels=1,
    feature_maps=1,
    kernel_height=2,
    kernel_width=2,
    v_stride=3,
    h_stride=3,
    optimized=True,
)
# cnn.add_Convolution2DLayer(
#     act_func=LRELU,
#     input_channels=1,
#     feature_maps=1,
#     kernel_height=3,
#     kernel_width=3,
# )
# cnn.add_PoolingLayer(kernel_height=2, kernel_width=2, pooling="max")

cnn.add_FlattenLayer()

cnn.add_FullyConnectedLayer(30, LRELU)

cnn.add_FullyConnectedLayer(20, LRELU)

cnn.add_OutputLayer(10, softmax)

scores = cnn.fit(
    x_train,
    y_train,
    lam=lam,
    batches=batches,
    epochs=epochs,
    X_val=x_val,
    t_val=y_val,
)

plt.plot(scores["train_acc"], label="Training")
plt.plot(scores["val_acc"], label="Validation")
plt.ylim([0.8, 1])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


prediction = cnn.predict(x_val)

print(accuracy(prediction, y_val))
print(confusion_matrix(np.argmax(y_val, axis=-1), np.argmax(prediction, axis=-1)))
plot_confusion_matrix(
    confusion_matrix(np.argmax(y_val, axis=-1), np.argmax(prediction, axis=-1)), labels
)

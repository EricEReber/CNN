from src.Layers import *
from src.FFNN import FFNN
import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt

np.random.seed(2023)

""" 
Test file for Conv-layers and Pooling-layers 
"""


def init_test():
    layer = Convolution2DLayer(3, 3, 2, 2, "same", lambda x: x, 2023)

    assert layer.kernel_tensor is not None


def forward_test(X):
    layer = Convolution2DLayer(
        input_channels=3,
        feature_maps=64,
        kernel_size=2,
        stride=2,
        pad="same",
        act_func=lambda x: x,
        seed=2023,
    )

    conv_rest = layer._feedforward(X)

    print(conv_rest.shape)
    assert conv_rest.shape == (X.shape[0], 64, X.shape[2], X.shape[3])

    plt.imshow(conv_rest[0, 0, :, :], vmin=0, vmax=255)
    plt.show()


def backward_test(X):
    layer = Convolution2DLayer(
        input_channels=3,
        feature_maps=64,
        kernel_size=2,
        stride=2,
        pad="same",
        act_func=lambda x: x,
        seed=2023,
    )

    # Shape of output we're testing with is (128, 128, 64, 3) -> (height, width, feature_maps, num_images)
    rand_grad = np.random.randn(128, 128, 64, 3)

    layer._backpropagate(X, rand_grad)


def max_pooling_test(X):
    pooling_layer = Pooling2DLayer(
        kernel_height=2,
        kernel_width=2,
        v_stride=2,
        h_stride=2,
        pooling="max",
    )

    mpl_img = pooling_layer._feedforward(X)
    # plt.imshow(mpl_img[0, 0, :, :], vmin=0, vmax=255, aspect="auto")
    # plt.show()


def max_pooling_back_test(X, delta):
    pooling_layer = Pooling2DLayer(
        kernel_height=2,
        kernel_width=2,
        v_stride=2,
        h_stride=2,
        pooling="max",
    )

    mpl_img = pooling_layer._feedforward(X)

    delta_grad = np.random.rand(
        mpl_img.shape[0], mpl_img.shape[1], mpl_img.shape[2], mpl_img.shape[3]
    )
    rv_mpl_img = pooling_layer._backpropagate(delta_grad, X)


def avr_pooling_test(X):
    pooling_layer = Pooling2DLayer(
        kernel_height=2,
        kernel_width=2,
        v_stride=2,
        h_stride=2,
        pooling="average",
    )

    avr_img = pooling_layer._feedforward(X)
    print(avr_img[0, 0, 0:4, 0:4])

    delta_grad = np.random.rand(
        avr_img.shape[0], avr_img.shape[1], avr_img.shape[2], avr_img.shape[3]
    )

    print(delta_grad[0, 0, 0:2, 0:2])
    rv_avr_img = pooling_layer._backpropagate(delta_grad)
    print(X[0, 0, 0:4, 0:4])
    print(rv_avr_img[0, 0, 0:4, 0:4])


if __name__ == "__main__":
    img_path = "/home/gregz/Files/CNN/data/luna.JPG"
    print(img_path)
    image = imageio.imread(img_path)
    images = np.ndarray((image.shape[0], image.shape[1], image.shape[2], 1))
    for i in range(1):
        images[:, :, :, i] = image[:, :, :]

    images = images.transpose(3, 2, 0, 1)
    # init_test()
    forward_test(images)

    # backward_test(images)
    # max_pooling_test(images)
    # max_pooling_back_test(images, images)

    # avr_pooling_test(images)

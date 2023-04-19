import math
import autograd.numpy as np
from copy import deepcopy, copy
from autograd import grad
from typing import Callable
from src.activationFunctions import *
from src.Schedulers import *
from src.costFunctions import *

"""
Interface contatining all the layers that will be available for construction of 
a models architecture.
"""

# global variables for readability
inputs = 0
nodes = 1
prev_nodes_inputs = 0
nodes = 1
bias = 1
batches = 0
input_channels = 1
feature_maps = 1
height = 2
width = 3


class Layer:
    def __init__(self, seed):
        self.seed = seed

    def _feedforward(self):
        raise NotImplementedError

    def _backpropagate(self):
        raise NotImplementedError

    def _reset_weights(self, previous_nodes):
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    # FullyConnectedLayer per default uses LRELU and Adam scheduler
    # with an eta of 0.0001, rho of 0.9 and rho2 of 0.999
    def __init__(
        self,
        nodes: int,
        act_func: Callable = LRELU,
        scheduler: Scheduler = Adam(eta=1e-4, rho=0.9, rho2=0.999),
        seed: int = None,
    ):
        super().__init__(seed)
        self.nodes = nodes
        self.act_func = act_func
        self.scheduler_weight = copy(scheduler)
        self.scheduler_bias = copy(scheduler)

        # initiate matrices for later
        self.weights = None
        self.a_matrix = None
        self.z_matrix = None

    def _feedforward(self, X_batch):
        # calculate z
        self.z_matrix = X_batch @ self.weights

        # calculate a, add bias
        bias = np.ones((X_batch.shape[inputs], 1)) * 0.01
        self.a_matrix = self.act_func(self.z_matrix)
        self.a_matrix = np.hstack([bias, self.a_matrix])

        # return a, the input for feedforward in next layer
        return self.a_matrix

    def _backpropagate(self, weights_next, delta_term_next, a_previous, lam):
        # take the derivative of the activation function
        activation_derivative = derivate(self.act_func)

        # calculate the delta term
        delta_term = (
            weights_next[bias:, :] @ delta_term_next.T
        ).T * activation_derivative(self.z_matrix)

        # intitiate matrix to store gradient
        # note that we exclude the bias term, which we will calculate later
        gradient_weights = np.zeros(
            (
                a_previous.shape[inputs],
                a_previous.shape[nodes] - bias,
                delta_term.shape[nodes],
            )
        )

        # calculate gradient = delta term * previous a
        for i in range(len(delta_term)):
            gradient_weights[i, :, :] = np.outer(a_previous[i, bias:], delta_term[i, :])

        # sum the gradient, divide by inputs
        gradient_weights = np.mean(gradient_weights, axis=inputs)
        # for the bias gradient we do not multiply by previous a
        gradient_bias = np.mean(delta_term, axis=inputs).reshape(
            1, delta_term.shape[nodes]
        )

        # regularization term
        gradient_weights += self.weights[bias:, :] * lam

        # send gradients into scheduler
        # returns update matrix which will be used to update the weights and bias
        update_matrix = np.vstack(
            [
                self.scheduler_bias.update_change(gradient_bias),
                self.scheduler_weight.update_change(gradient_weights),
            ]
        )

        # update weights
        self.weights -= update_matrix

        # return weights and delta term, input for backpropagation in previous layer
        return self.weights, delta_term

    def _reset_weights(self, previous_nodes):
        # sets seed to remove randomness inbetween runs
        if self.seed is not None:
            np.random.seed(self.seed)

        # add bias, initiate random weights
        self.weights = np.random.randn(previous_nodes + bias, self.nodes)

        # returns number of nodes, used for reset_weights in next layer
        return self.nodes

    def _reset_scheduler(self):
        # resets scheduler per epoch
        self.scheduler_weight.reset()
        self.scheduler_bias.reset()

    def get_prev_a(self):
        # returns a matrix, used in backpropagation
        return self.a_matrix


class OutputLayer(FullyConnectedLayer):
    def __init__(
        self,
        nodes: int,
        output_func: Callable = LRELU,
        cost_func: Callable = CostCrossEntropy,
        scheduler: Scheduler = Adam(eta=1e-4, rho=0.9, rho2=0.999),
        seed: int = None,
    ):
        super().__init__(nodes, output_func, copy(scheduler), seed)
        self.cost_func = cost_func

        # initiate matrices for later
        self.weights = None
        self.a_matrix = None
        self.z_matrix = None

        # decides if the output layer performs binary or multi-class classification
        self.set_pred_format()

    def _feedforward(self, X_batch: np.ndarray):

        # calculate a, z
        # note that bias is not added as this would create an extra output class
        self.z_matrix = X_batch @ self.weights
        self.a_matrix = self.act_func(self.z_matrix)

        # returns prediction
        return self.a_matrix

    def _backpropagate(self, target, a_previous, lam):
        # note that in the OutputLayer the activation function is the output function
        activation_derivative = derivate(self.act_func)

        # calculate output delta terms
        # for multi-class or binary classification
        if self.pred_format == "Multi-class":
            delta_term = self.a_matrix - target
        else:
            cost_func_derivative = grad(self.cost_func(target))
            delta_term = activation_derivative(self.z_matrix) * cost_func_derivative(
                self.a_matrix
            )

        # intiate matrix that stores gradient
        gradient_weights = np.zeros(
            (
                a_previous[:, 1:].shape[0],
                a_previous[:, 1:].shape[1],
                delta_term.shape[1],
            )
        )

        # calculate gradient = delta term * previous a
        for i in range(len(delta_term)):
            gradient_weights[i, :, :] = np.outer(a_previous[i, 1:], delta_term[i, :])

        # sum the gradient, divide by inputs
        gradient_weights = np.mean(gradient_weights, axis=0)
        # for the bias gradient we do not multiply by previous a
        gradient_bias = np.mean(delta_term, axis=0).reshape(1, delta_term.shape[1])

        # regularization term
        gradient_weights += self.weights[1:, :] * lam

        # send gradients into scheduler
        # returns update matrix which will be used to update the weights and bias
        update_matrix = np.vstack(
            [
                self.scheduler_bias.update_change(gradient_bias),
                self.scheduler_weight.update_change(gradient_weights),
            ]
        )

        # update weights
        self.weights -= update_matrix

        # return weights and delta term, input for backpropagation in previous layer
        return self.weights, delta_term

    def set_pred_format(self):
        # sets prediction format to either regression, binary or multi-class classification
        if self.act_func.__name__ is None or self.act_func.__name__ == "identity":
            self.pred_format = "Regression"
        elif self.act_func.__name__ == "sigmoid" or self.act_func.__name__ == "tanh":
            self.pred_format = "Binary"
        else:
            self.pred_format = "Multi-class"

    def _reset_weights(self, previous_nodes):
        # sets seed to remove randomness inbetween runs
        if self.seed is not None:
            np.random.seed(self.seed)

        # add bias, initiate random weights
        bias = 1
        self.weights = np.random.rand(previous_nodes + bias, self.nodes)

        # returns number of nodes, used for reset_weights in next layer
        return self.nodes

    def _reset_scheduler(self):
        # resets scheduler per epoch
        self.scheduler_weight.reset()
        self.scheduler_bias.reset()

    def get_pred_format(self):
        # returns format of prediction
        return self.pred_format


class Convolution2DLayer(Layer):
    def __init__(
        self,
        input_channels,
        feature_maps,
        kernel_height,
        kernel_width,
        v_stride,
        h_stride,
        pad,
        act_func: Callable,
        seed=None,
        reset_self=True,
    ):
        super().__init__(seed)
        self.input_channels = input_channels
        self.feature_maps = feature_maps
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.v_stride = v_stride
        self.h_stride = h_stride
        self.pad = pad
        self.act_func = act_func

        # such that the layer can be used on its own
        # outside of the CNN module
        if reset_self == True:
            self._reset_weights_single_layer()

    def _reset_weights_single_layer(self):
        # sets seed to remove randomness inbetween runs
        if self.seed is not None:
            np.random.seed(self.seed)

        # initializes kernel matrix
        self.kernel = np.ndarray(
            (
                self.input_channels,
                self.feature_maps,
                self.kernel_height,
                self.kernel_width,
            )
        )

        # randomly initializes weights
        for i in range(self.kernel.shape[0]):
            for j in range(self.kernel.shape[1]):
                self.kernel[i, j, :, :] = np.random.rand(
                    self.kernel_height, self.kernel_width
                )

    def _reset_weights(self, previous_nodes):
        # sets weights
        self._reset_weights_single_layer()

        # returns shape of output used for subsequent layer's weight initiation
        strided_height = int(np.ceil(previous_nodes.shape[height] / self.v_stride))
        strided_width = int(np.ceil(previous_nodes.shape[width] / self.h_stride))
        next_nodes = np.ones(
            (previous_nodes.shape[inputs], self.feature_maps, strided_height, strided_width)
        )
        return next_nodes / self.kernel_height

    def _feedforward(self, X_batch):
        # note that the shape of X_batch = [batch_size, input_maps, img_height, img_width]

        # pad the input batch
        X_batch_padded = self._padding(X_batch)

        # calculate height and width after stride
        strided_height = int(np.ceil(X_batch.shape[height] / self.v_stride))
        strided_width = int(np.ceil(X_batch.shape[width] / self.h_stride))

        # create output array
        output = np.ndarray(
            (
                X_batch.shape[0],
                self.feature_maps,
                strided_height,
                strided_width,
            )
        )

        # save input and output for backpropagation
        self.input = X_batch
        self.output_shape = output.shape

        # checking for errors, no need to look here :)
        self._check_for_errors()

        # convolve input with kernel
        for img in range(X_batch.shape[inputs]):
            for chin in range(self.input_channels):
                for fmap in range(self.feature_maps):
                    new_x = 0
                    for x in range(0, X_batch.shape[height], self.v_stride):
                        new_y = 0
                        for y in range(0, X_batch.shape[width], self.h_stride):
                            output[img, fmap, new_x, new_y] = np.sum(
                                X_batch_padded[
                                    img,
                                    chin,
                                    x : x + self.kernel_height,
                                    y : y + self.kernel_width,
                                ]
                                * self.kernel[chin, fmap, :, :]
                            )
                            new_y += 1
                        new_x += 1

        # Pay attention to the fact that we're not rotating the kernel by 180 degrees when filtering the image in
        # the convolutional layer, as convolution in terms of Machine Learning is a procedure known as cross-correlation
        # in image processing and signal processing

        # return a
        return self.act_func(output / (self.kernel_height))

    def _backpropagate(self, delta_term_next):
        # intiate matrices
        delta_term = np.zeros((self.input.shape))
        kernel_grad = np.zeros((self.kernel.shape))

        # pad input for convolution
        X_batch_padded = self._padding(self.input)

        # Since an activation function is used at the output of the convolution layer, its derivative
        # has to be accounted for in the backpropagation -> as if ReLU was a layer on its own.
        act_derivative = derivate(self.act_func)
        delta_term_next = act_derivative(delta_term_next)

        # reconstruct shape
        # TODO does this work?
        if self.v_stride > 1 or self.h_stride > 1:
            v_ind = 1
            h_ind = 1
            for i in range(delta_term_next.shape[height]):
                for j in range(self.h_stride - 1):
                    delta_term_next = np.insert(delta_term_next, h_ind, 0, axis=height)
                for k in range(self.v_stride - 1):
                    delta_term_next = np.insert(delta_term_next, v_ind, 0, axis=width)
                v_ind += self.v_stride
                h_ind += self.h_stride

            delta_term_next = delta_term_next[
                :, :, : self.input.shape[height], : self.input.shape[width]
            ]

        # The gradient received from the next layer also needs to be padded
        delta_term_next = self._padding(delta_term_next)

        for img in range(self.input.shape[inputs]):
            for chin in range(self.input_channels):
                for fmap in range(self.feature_maps):
                    for x in range(self.input.shape[height]):
                        for y in range(self.input.shape[width]):
                            delta_term[img, chin, x, y] = np.sum(
                                delta_term_next[
                                    img,
                                    fmap,
                                    x : x + self.kernel_height,
                                    y : y + self.kernel_width,
                                ]
                                * np.rot90(np.rot90(self.kernel[chin, fmap, :, :]))
                            )

        for chin in range(self.input_channels):
            for fmap in range(self.feature_maps):
                for k_x in range(self.kernel_height):
                    for k_y in range(self.kernel_width):
                        kernel_grad[chin, fmap, k_x, k_y] = np.sum(
                            X_batch_padded[
                                img,
                                chin,
                                x : x + self.kernel_height,
                                y : y + self.kernel_width,
                            ]
                            * delta_term_next[
                                img,
                                fmap,
                                x : x + self.kernel_height,
                                y : y + self.kernel_width,
                            ]
                        )
                        # Each filter is updated
        self.kernel[:, :, :, :] -= kernel_grad[:, :, :, :]

        return delta_term

    def _padding(self, X_batch, batch_type="image"):
        # TODO: Need fixing to output so the channels are merged back together after padding is finished!

        if self.pad == "same" and batch_type == "image":
            padded_height = X_batch.shape[height] + (self.kernel_height // 2) * 2
            padded_width = X_batch.shape[width] + (self.kernel_width // 2) * 2
            half_kernel_height = self.kernel_height // 2
            half_kernel_width = self.kernel_width // 2

            new_tensor = np.ndarray(
                (X_batch.shape[inputs], X_batch.shape[feature_maps], padded_height, padded_width)
            )

            for img in range(X_batch.shape[inputs]):
                padded_img = np.zeros((X_batch.shape[feature_maps], padded_height, padded_width))
                padded_img[
                    :,
                    half_kernel_height : padded_height - half_kernel_height,
                    half_kernel_width : padded_width - half_kernel_width,
                ] = X_batch[img, :, :, :]
                new_tensor[img, :, :, :] = padded_img[:, :, :]

            return new_tensor

        elif self.pad == "same" and batch_type == "grad":
            padded_height = X_batch.shape[2] + (self.kernel_height // 2) * 2
            padded_width = X_batch.shape[3] + (self.kernel_width // 2) * 2
            half_kernel_height = self.kernel_height // 2
            half_kernel_width = self.kernel_width // 2

            new_tensor = np.zeros(
                (X_batch.shape[0], X_batch.shape[1], padded_height, padded_width)
            )

            new_tensor[:, :, : X_batch.shape[2], : X_batch.shape[3]] = X_batch[
                :, :, :, :
            ]

            return new_tensor

        else:
            return X_batch

    def _check_for_errors(self):
        if self.input.shape[1] != self.input_channels:
            raise AssertionError(
                f"ERROR: Number of input channels in data ({self.input.shape[1]}) is not equal to input channels in Convolution2DLayerOPT ({self.input_channels})! Please change the number of input channels of the Convolution2DLayer such that they are equal"
            )


class Convolution2DLayerOPT(Convolution2DLayer):
    def __init__(
        self,
        input_channels,  # number of maps the input is split into
        feature_maps,  # also known as feature maps
        kernel_height,
        kernel_width,
        v_stride,
        h_stride,
        pad,
        act_func: Callable,
        seed=None,
        reset_self=True,
    ):
        super().__init__(
            input_channels,
            feature_maps,
            kernel_height,
            kernel_width,
            v_stride,
            h_stride,
            pad,
            act_func,
            seed,
        )
        if reset_self == True:
            self._reset_weights_single_layer()

    def _extract_windows(self, X_batch, batch_type="image"):
        # TODO: Change padding so that it takes the height and width of kernel as arguments

        windows = []
        if batch_type == "image":
            # pad the images
            X_batch_padded = self._padding(X_batch, batch_type="image")
            img_height, img_width = X_batch_padded.shape[2:]
            # For each location in the image...
            for h in range(
                0,
                img_height - self.kernel_height + self.kernel_height % 2,
                self.v_stride,
            ):
                for w in range(
                    0,
                    img_width - self.kernel_width + self.kernel_width % 2,
                    self.h_stride,
                ):
                    # ...get an image patch of size [fil_size, fil_size]

                    window = X_batch_padded[
                        :,
                        :,
                        h : h + self.kernel_height,
                        w : w + self.kernel_width,
                    ]
                    windows.append(window)
            return np.stack(windows)

        if batch_type == "grad":
            # TODO description
            if self.v_stride < 2 or self.v_stride % 2 == 0:
                v_stride = 0
            else:
                v_stride = int(np.floor(self.v_stride / 2))

            if self.h_stride < 2 or self.h_stride % 2 == 0:
                h_stride = 0
            else:
                h_stride = int(np.floor(self.h_stride / 2))

            upsampled_height = (X_batch.shape[height] * self.v_stride) - v_stride

            upsampled_width = (X_batch.shape[width] * self.h_stride) - h_stride

            ind = 1
            # TODO need description of what this does. Why for range of width? What about height?
            for i in range(X_batch.shape[height]):
                for j in range(self.h_stride - 1):
                    X_batch = np.insert(X_batch, ind, 0, axis=height)
                for k in range(self.v_stride - 1):
                    X_batch = np.insert(X_batch, ind, 0, axis=width)
                ind += self.v_stride

            X_batch = X_batch[:, :, :upsampled_height, :upsampled_width]

            X_batch_padded = self._padding(X_batch, batch_type="grad")

            windows = []
            for h in range(
                X_batch_padded.shape[height] - self.kernel_height + self.kernel_width % 2
            ):
                for w in range(
                    X_batch_padded.shape[width] - self.kernel_width + self.kernel_width % 2
                ):
                    # ...get an image patch of size [fil_size, fil_size]

                    window = X_batch_padded[
                        :, :, h : h + self.kernel_height, w : w + self.kernel_width
                    ]
                    windows.append(window)

            return np.stack(windows), upsampled_height, upsampled_width

    def _feedforward(self, X_batch):
        # The optimized _feedforward method is difficult to understand but computationally more efficient
        # for a more "by the book" approach, please look at the _feedforward method of Convolution2DLayer

        # save the input for backpropagation
        self.input = X_batch

        # check that there are the correct amount of input channels
        self._check_for_errors()

        # calculate new shape after stride
        strided_height = int(np.ceil(X_batch.shape[height] / self.v_stride))
        strided_width = int(np.ceil(X_batch.shape[width] / self.h_stride))

        # get windows of the image for more computationally efficient convolution
        windows = self._extract_windows(X_batch)
        windows = windows.transpose(1, 0, 2, 3, 4).reshape(
            X_batch.shape[0],
            strided_height * strided_width,
            -1,
        )

        # reshape the kernel for more computationally efficient convolution
        kernel = self.kernel
        kernel = kernel.transpose(0, 2, 3, 1).reshape(
            kernel.shape[0] * kernel.shape[2] * kernel.shape[3],
            -1,
        )

        # use simple matrix calculation to obtain output
        output = (
            (windows @ kernel)
            .reshape(
                X_batch.shape[inputs],
                strided_height,
                strided_width,
                -1,
            )
            .transpose(0, 3, 1, 2)
        )

        # The output is reshaped and rearranged to appropriate shape
        return self.act_func(
            output / (self.kernel_height * X_batch.shape[feature_maps])
        )

    def _backpropagate(self, delta_term_next):
        # The optimized _backpropagate method is difficult to understand but computationally more efficient
        # for a more "by the book" approach, please look at the _backpropagate method of Convolution2DLayer
        act_derivative = derivate(self.act_func)
        delta_term_next = act_derivative(delta_term_next)

        strided_height = int(np.ceil(self.input.shape[height] / self.v_stride))
        strided_width = int(np.ceil(self.input.shape[width] / self.h_stride))
        kernel = self.kernel

        windows = self._extract_windows(self.input, "image").reshape(
            self.input.shape[inputs] * strided_height * strided_width, -1
        )

        output_grad_tr = delta_term_next.transpose(0, 2, 3, 1).reshape(
            self.input.shape[inputs] * strided_height * strided_width, -1
        )

        kernel_grad = (
            (windows.T @ output_grad_tr)
            # TODO change kernel indices with global variables
            .reshape(kernel.shape[0], kernel.shape[2], kernel.shape[3], kernel.shape[1])
            .transpose(0, 3, 1, 2)
        )

        # Computing the input gradient
        windows_out, upsampled_height, upsampled_width = self._extract_windows(
            delta_term_next, "grad"
        )

        # print(f"{windows_out.transpose(1, 0, 2, 3, 4).shape=}")
        # print(f"{self.input.shape[0]=}")
        # print(f"{upsampled_height=}")
        # print(f"{upsampled_width=}")
        # TODO this line causes a crash if kernel size has one odd number and one whole number
        # (other asymmetric kernels work)
        # TODO switch out (1, 0, 2, 3, 4) with global variables
        windows_out = windows_out.transpose(1, 0, 2, 3, 4).reshape(
            self.input.shape[0] * upsampled_height * upsampled_width,
            -1,
        )
        kernel_r = kernel.reshape(self.input_channels, -1)

        input_grad = (windows_out @ kernel_r.T).reshape(
            self.input.shape[0], upsampled_height, upsampled_width, kernel.shape[0]
        )
        input_grad = input_grad.transpose(0, 3, 1, 2)

        # Update the weights in the kernel
        self.kernel -= kernel_grad

        # Output the gradient to propagate backwards
        return input_grad

    def _check_for_errors(self):
        # compares input channels of data to input channels of Convolution2DLayer
        if self.input.shape[input_channels] != self.input_channels:
            raise AssertionError(
                f"ERROR: Number of input channels in data ({self.input.shape[1]}) is not equal to input channels in Convolution2DLayerOPT ({self.input_channels})! Please change the number of input channels of the Convolution2DLayer such that they are equal"
            )


class Pooling2DLayer(Layer):
    def __init__(
        self,
        kernel_height,
        kernel_width,
        v_stride,
        h_stride,
        pooling="max",
        seed=None,
    ):
        super().__init__(seed)
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.v_stride = v_stride
        self.h_stride = h_stride
        self.pooling = pooling

    def _feedforward(self, X_batch):
        # Saving the input for use in the backwardpass
        self.input = X_batch

        # check if user is silly
        self._check_for_errors()

        # Computing the size of the feature maps based on kernel size and the stride parameter
        strided_height = (X_batch.shape[height] - self.kernel_height) // self.v_stride + 1
        if X_batch.shape[height] == X_batch.shape[height]:
            strided_width = strided_height
        else:
            strided_width = (X_batch.shape[width] - self.kernel_width) // self.h_stride + 1

        output = np.ndarray(
            (X_batch.shape[inputs], X_batch.shape[feature_maps], strided_height, strided_width)
        )

        if self.pooling == "max":
            self.pooling_action = np.max
        elif self.pooling == "average":
            self.pooling_action = np.mean

        for img in range(output.shape[inputs]):
            for fmap in range(output.shape[feature_maps]):
                for x in range(strided_height):
                    for y in range(strided_width):
                        output[img, fmap, x, y] = self.pooling_action(
                            X_batch[
                                img,
                                fmap,
                                (x * self.v_stride) : (x * self.v_stride)
                                + self.kernel_height,
                                (y * self.h_stride) : (y * self.h_stride)
                                + self.kernel_width,
                            ]
                        )

        return output

    def _backpropagate(self, delta_term_next):
        delta_term = np.zeros((self.input.shape))

        for img in range(delta_term_next.shape[inputs]):
            for fmap in range(delta_term_next.shape[feature_maps]):
                for x in range(0, delta_term_next.shape[height], self.v_stride):
                    for y in range(0, delta_term_next.shape[width], self.h_stride):
                        if self.pooling == "max":
                            window = self.input[
                                img,
                                fmap,
                                x : x + self.kernel_height,
                                y : y + self.kernel_width,
                            ]

                            i, j = np.unravel_index(window.argmax(), window.shape)

                            delta_term[
                                img,
                                fmap,
                                (x + i),
                                (y + j),
                            ] += delta_term_next[img, fmap, x, y]

                        if self.pooling == "average":
                            delta_term[
                                img,
                                fmap,
                                x : x + self.kernel_height,
                                y : y + self.kernel_width,
                            ] = (
                                delta_term_next[img, fmap, x, y]
                                / self.kernel_height
                                / self.kernel_width
                            )
        return delta_term

    def _reset_weights(self, previous_nodes):
        strided_height = (
            previous_nodes.shape[height] - self.kernel_height
        ) // self.v_stride + 1
        if previous_nodes.shape[height] == previous_nodes.shape[width]:
            strided_width = strided_height
        else:
            strided_width = (
                previous_nodes.shape[width] - self.kernel_width
            ) // self.h_stride + 1

        output = np.ones(
            (
                previous_nodes.shape[inputs],
                previous_nodes.shape[feature_maps],
                strided_height,
                strided_width,
            )
        )
        return output

    def _check_for_errors(self):
        assert (
            self.input.shape[width] >= self.kernel_width
        ), f"ERROR: Pooling kernel width ({self.kernel_width}) larger than data width ({self.input.shape[2]}), please lower the kernel width of the Pooling2DLayer"
        assert (
            self.input.shape[height] >= self.kernel_height
        ), f"ERROR: Pooling kernel height ({self.kernel_height}) larger than data height ({self.input.shape[3]}), please lower the kernel height of the Pooling2DLayer"


class FlattenLayer(Layer):
    def __init__(self, act_func=LRELU, seed=None):
        super().__init__(seed)
        self.act_func = act_func

    def _feedforward(self, X_batch):
        # save input for backpropagation
        self.input_shape = X_batch.shape
        # Remember, the data has the following shape: (B, FM, H, W, ) in the convolutional layers
        # whilst the data has the shape (B, FM * H * W) in the fully connected layers
        # FM = Feature maps, B = Batch size, H = Height and W = Width
        X_batch = X_batch.reshape(
            X_batch.shape[batches],
            X_batch.shape[feature_maps] * X_batch.shape[height] * X_batch.shape[width],
        )

        # add bias to a
        self.z_matrix = X_batch
        bias = np.ones((X_batch.shape[inputs], 1)) * 0.01
        self.a_matrix = np.hstack([bias, X_batch])

        return self.a_matrix

    def _backpropagate(self, weights_next, delta_term_next):
        # calculates delta_term_next and reshapes it for convolutional layers
        # FlattenLayer does not update weights
        activation_derivative = derivate(self.act_func)

        delta_term = (
            weights_next[bias:, :] @ delta_term_next.T
        ).T * activation_derivative(self.z_matrix)

        return delta_term.reshape(self.input_shape)

    def get_prev_a(self):
        return self.a_matrix

    def _reset_weights(self, previous_nodes):
        # note that the previous nodes to the FlattenLayer are from the convolutional layers
        previous_nodes = previous_nodes.reshape(
            previous_nodes.shape[inputs],
            previous_nodes.shape[feature_maps]
            * previous_nodes.shape[height]
            * previous_nodes.shape[width],
        )
        return previous_nodes.shape[nodes]

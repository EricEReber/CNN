import math
import autograd.numpy as np
from copy import deepcopy, copy
from autograd import grad
from typing import Callable
from src.activationFunctions import *
from src.Schedulers import *

"""
Interface contatining all the layers that will be available for construction of 
a models architecture.
"""
# TODO: Start implementation of FullyConnected, Output and Convolution Layer


class Layer:
    def __init__(self, seed):
        self.seed = seed

    def _feedforward(self):
        raise NotImplementedError

    def _backpropagate(self):
        raise NotImplementedError

    def _reset_weights(self):
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    def __init__(
        self,
        nodes,
        act_func: Callable,
        scheduler: Scheduler,
        seed=None,
    ):
        super().__init__(seed)
        self.nodes = nodes
        self.act_func = act_func
        self.scheduler_weight = copy(scheduler)
        self.scheduler_bias = copy(scheduler)

        self.weights = None
        self.a_matrix = None
        self.z_matrix = None

        self._reset_weights()

    def _feedforward(self, X):

        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # Adding bias to the data
        bias = np.ones((X.shape[0], 1)) * 0.01
        X = np.hstack([bias, X])

        self.z_matrix = X @ self.weights
        self.a_matrix = self.act_func(self.z_matrix)

        return self.a_matrix

    def _backpropagate(self, weights_next, delta_next, lam):

        activation_derivative = derivate(self.act_func)

        delta_matrix = (weights_next[1:, :] @ delta_next.T).T * activation_derivative(
            self.z_matrix
        )

        gradient_weights = np.zeros(
            (
                self.a_matrix[:, 1:].shape[0],
                self.a_matrix[:, 1:].shape[1],
                delta_matrix.shape[1],
            )
        )

        for i in range(len(delta_matrix)):
            gradient_weights[i, :, :] = np.outer(
                self.a_matrix[i, 1:], delta_matrix[i, :]
            )

        gradient_weights = np.mean(gradient_weights, axis=0)
        gradient_bias = np.mean(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])

        # regularization term
        gradient_matrix += self.weights[1:, :] * lam

        # TODO: This part needs changing! Scheduler should update weights and bias simultaneously,
        # and not require two instances of the same class for the weight and bias update task
        update_matrix = np.vstack(
            [
                self.scheduler_weight.update_change(gradient_matrix),
                self.scheduler_bias.update_change(gradient_bias),
            ]
        )

        self.weights -= update_matrix

        return self.weights, delta_matrix

    def _reset_weights(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = np.random.randn(self.nodes[0] + 1, self.nodes[1])

    def _reset_scheduler(self):
        self.scheduler_weight.reset()
        self.scheduler_bias.reset()


# TODO: Test if OutputLayer inherits the constructor in the desired way, or if some changes need to be applied.
class OutputLayer(FullyConnectedLayer):
    def __init__(
        self,
        nodes: tuple[int],
        output_func: Callable,
        cost_func: Callable,
        scheduler: Scheduler,
        seed=None,
    ):
        super().__init__(nodes, output_func, copy(scheduler), seed)
        self.cost_func = cost_func

        self.weights = None
        self.a_matrix = None
        self.z_matrix = None

        self._reset_weights()
        self.set_prediction()  # Decides what type of prediction the output layer performs

    def _feedforward(self, X: np.ndarray):

        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # Adding bias to the data
        bias = np.ones((X.shape[0], 1)) * 0.01
        X = np.hstack([bias, X])

        self.z_matrix = X @ self.weights
        self.a_matrix = self.act_func(self.z_matrix)

        return self.a_matrix

    def _backpropagate(self, target, lam):

        print(f"{self.a_matrix.shape=}")
        # Again, remember that in the OutputLayer the activation function is the output function
        activation_derivative = derivate(self.act_func)

        if self.prediction == "Multi-class":
            delta_matrix = self.a_matrix - target
        else:
            cost_func_derivative = grad(self.cost_func(target))
            delta_matrix = activation_derivative(self.z_matrix) * cost_func_derivative(
                self.a_matrix
            )

        gradient_weights = np.zeros(
            (
                self.a_matrix[:, 1:].shape[0],
                self.a_matrix[:, 1:].shape[1],
                delta_matrix.shape[1],
            )
        )

        for i in range(len(delta_matrix)):
            gradient_weights[i, :, :] = np.outer(
                self.a_matrix[i, 1:], delta_matrix[i, :]
            )

        gradient_weights = np.mean(gradient_weights, axis=0)
        gradient_bias = np.mean(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])

        # regularization term
        print(f"{gradient_weights.shape=}")
        print(f"{self.weights[1:, :].shape=}")
        gradient_weights += self.weights[1:, :] * lam

        # TODO: This part needs changing! Scheduler should update weights and bias simultaneously,
        # and not require two instances of the same class for the weight and bias update task
        update_matrix = np.vstack(
            [
                self.scheduler_weight.update_change(gradient_matrix),
                self.scheduler_bias.update_change(gradient_bias),
            ]
        )

        self.weights -= update_matrix

        return self.weights, delta_matrix

    def predict(self, X: np.ndarray, *, threshold=0.5):

        predict = self._feedforward(X)

        if self.prediction == "Binary":
            return np.where(predict > threshold, 1, 0)
        else:
            return predict

    def set_prediction(self):

        if self.act_func.__name__ is None:
            self.prediction = "Regression"
        elif self.act_func.__name__ == "sigmoid" or self.act_func.__name__ == "tanh":
            self.prediction = "Binary"
        else:
            self.prediction = "Mulit-class"

    def _reset_weights(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        bias = 1
        self.weights = np.random.rand(self.nodes[0] + bias, self.nodes[1])

    def _reset_scheduler(self):
        self.scheduler_weight.reset()
        self.scheduler_bias.reset()


class Convolution2DLayer(Layer):
    def __init__(
        self,
        input_channels,  # number of maps the input is split into
        feature_maps,  # also known as feature maps
        kernel_size,
        stride,
        pad,
        act_func: Callable,
        seed=None,
    ):
        super().__init__(seed)
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.feature_maps = feature_maps
        self.stride = stride
        self.act_func = act_func
        self.pad = pad

        self._reset_weights()

    def _reset_weights(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        self.kernel_tensor = np.ndarray(
            (self.input_channels, self.feature_maps, self.kernel_size, self.kernel_size)
        )

        for i in range(self.kernel_tensor.shape[0]):
            for j in range(self.kernel_tensor.shape[1]):
                self.kernel_tensor[i, j, :, :] = np.random.rand(
                    self.kernel_size, self.kernel_size
                )

    def _feedforward(self, X):
        
        X_pad = self._padding(X)

        input = self._padding(X)

        output = np.ndarray((X.shape[0], X.shape[1], self.feature_maps, X.shape[3]))

        # Will need this parameter for backpropagation
        self.output_shape = output.shape
        
        start = self.kernel_size//2
        # if self.kernel_size % 2 != 0: 
        #     end = start + 1 
        # else: 
        end = start 

        for img in range(X.shape[3]): 
            for chin in range(self.input_channels): 
                for chout in range(self.feature_maps): 
                    for x in range(start, X.shape[0]+end, self.stride): 
                        for y in range(start, X.shape[1]+end, self.stride): 

                            output[x-start, y-start, chout, img] = \
                                np.sum(X_pad[x - start : x+end + end, y - start : y + end, chin, img] 
                                * self.kernel_tensor[chin, chout, :, :])
                            

        start = self.kernel_size // 2
        if self.kernel_size % 2 != 0:
            end = start + 1
        else:
            end = start

        for img in range(input.shape[3]):
            for chin in range(self.input_channels):
                for chout in range(self.feature_maps):
                    for x in range(start, input.shape[0], self.stride):
                        for y in range(start, input.shape[1], self.stride):

                            output[x - start, y - start, chout, img] = np.sum(
                                input[
                                    x - start : x + end, y - start : y + end, chin, img
                                ]
                                * self.kernel_tensor[chin, chout, :, :]
                            )

                            # Can also be written in a less intuitive way by introducing two extra for loops:
                            """
                            for k_x in range(self.kernel_size): 
                                for k_y in range(self.kernel_size): 
                                    output[x, y, chout, img] += self.kernel_tensor[chin, chout, kx, ky] * input[x+k_x, y+k_y, chin, img]
                            """
                            # Pay attention to the fact that we're not rotating the kernel by 180 degrees when filtering the image in
                            # the convolutional layer, as convolution in terms of Machine Learning is a procedure known as cross-correlation
                            # in image processing and signal processing

        return self.act_func(output)

    def _backpropagate(self, X, delta_next):

        delta = np.zeros((X.shape))
        kernel_grad = np.zeros((self.kernel_tensor.shape))

        X_pad = self._padding(X)

        # Since an activation function is used at the output of the convolution layer, its derivative
        # has to be accounted for in the backpropagation -> as if ReLU a layer on its own.
        act_derivative = derivate(self.act_func)
        delta_next = act_derivative(delta_next)

        # The gradient received from the next layer also needs to be padded
        delta_next = self._padding(delta_next)

        start = self.kernel_size // 2
        if self.kernel_size % 2 != 0:
            end = start + 1
        else:
            end = start

        for img in range(X.shape[3]):
            for chin in range(self.input_channels):
                for chout in range(self.feature_maps):
                    for x in range(start, X.shape[0]+end, self.stride):
                        for y in range(start, X.shape[1]+end, self.stride):
                            
                            print(f'computing x:{x}, y:{y}')
                            delta[x, y, chin, img] = np.sum(
                                delta_next[
                                    x - start : x + end, y - start : y + end, chout, img
                                ]
                                * np.rot90(
                                    np.rot90(self.kernel_tensor[chin, chout, :, :])
                                )
                            )

                            for k_x in range(self.kernel_size):
                                for k_y in range(self.kernel_size):

                                    kernel_grad[chin, chout, k_x, k_y] = np.sum(
                                        X_pad[
                                            x - start : x + end,
                                            y - start : y + end,
                                            chin,
                                            img,
                                        ]
                                        * delta_next[
                                            x - start : x + end,
                                            y - start : y + end,
                                            chout,
                                            img,
                                        ]
                                    )
                                    print(f'update{img}')
                                    # Each filter is updated
                                    self.kernel_tensor[chin, chout, :, :] -= kernel_grad[chin, chout,:,:]

        return delta

    def _padding(self, batch):

        # TODO: Need fixing to output so the channels are merged back together after padding is finished!

        if self.pad == 'same':
            print(batch.shape)
            new_height = batch[:,:,0,0].shape[0] + (self.kernel_size//2)*2
            new_width = batch[:,:,0,0].shape[1] + (self.kernel_size//2)*2
            k_height = self.kernel_size//2

        if self.pad == "same":

            new_height = batch[:, :, 0, 0].shape[0] + (self.kernel_size // 2) * 2
            new_width = batch[:, :, 0, 0].shape[1] + (self.kernel_size // 2) * 2
            k_height = self.kernel_size // 2

            new_tensor = np.ndarray(
                (new_height, new_width, batch.shape[2], batch.shape[3])
            )

            for img in range(batch.shape[3]):

                padded_img = np.zeros(
                    (new_height, new_width, batch[:, :, :, img].shape[2])
                )
                padded_img[
                    k_height : new_height - k_height, k_height : new_width - k_height, :
                ] = batch[:, :, :, img]
                new_tensor[:, :, :, img] = padded_img[:, :, :]

            return new_tensor

        else:
            return batch


class Pooling2DLayer(Layer):
    def __init__(self, seed, kernel_size, stride, pooling="max"):
        super().__init__(seed)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling = pooling

    def _feedforward(self, X):

        # Computing the size of the feature maps based on kernel size and the stride parameter
        new_height = (X[:, :, 0, 0].shape[0] - self.kernel_size) / self.stride + 1
        if X[:, :, 0, 0].shape[1] == X[:, :, 0, 0].shape[0]:
            new_width = new_height
        else:
            new_width = (X[:, :, 0, 0].shape[1] - self.kernel_size) / self.stride + 1

        output = np.ndarray((new_height, new_width, X.shape[2], X.shape[3]))

        if self.pooling == "max":
            self.pooling_action = np.max
        else:
            self.pooling_action = np.mean

        for img in range(X.shape[3]):
            for map in range(X.shape[2]):
                new_x, new_y = 0, 0
                for x in range(0, X.shape[0], self.stride):
                    for y in range(0, X.shape[1], self.stride):

                        output[new_x, new_y] = self.pooling_action(
                            X[
                                x : x + self.kernel_size,
                                y : y + self.kernel_size,
                                map,
                                img,
                            ]
                        )
                    new_y += 1
                new_x += 1

        return output


class FlattenLayer(Layer):
    def __init__(self, seed):
        super().__init__(seed)

    def _feedforward(self, X):

        self.input_shape = X.shape
        # Remember, the data has the following shape: (H, W, FM, B) Where H = Height, W = Width, FM = Feature maps and B = Batch size
        return X.reshape(-1, X.shape[3])

    def _backpropagate(self, delta_next):

        return delta_next.reshape(self.input_shape)

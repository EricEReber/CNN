import math
import autograd.numpy as np
from autograd import grad 
import sys
from typing import Callable 
from Schedulers import *
from activation_funcs import * 

"""
Interface contatining all the layers that will be available for construction of 
a models architecture.
"""
# TODO: Start implementation of FullyConnected, Output and Convolution Layer

class Layer: 

    def __init__(self): 
        self
    
    def _feedforward(self): 
        raise NotImplementedError 

    def _backpropagate(self): 
        raise NotImplementedError

    def _reset_weights(self): 
        raise NotImplementedError

    def _initialize_weights(self): 
        raise NotImplementedError

class FullyConnectedLayer(Layer): 
    
    def __init__(
            self, 
            nodes: tuple[int], 
            act_func: Callable,  
            scheduler: Scheduler,
            *scheduler_args: list, 
            seed=None,
    ):

        self.nodes = nodes
        self.act_func = act_func
        self.scheduler_weight = scheduler(*scheduler_args)
        self.scheduler_bias = scheduler(*scheduler_args)
        self.weights = self._initialize_weights()
        self.seed = seed

    def _initialize_weights(self):

        if self.seed is not None: 
            np.random.seed(self.seed)

        weight_matrix = np.random.rand(self.nodes[0]+1, self.nodes[1])
        
        return weight_matrix

    def _reset_weights(self): 
        
        if self.seed is not None: 
            np.random.seed(self.seed)

        self.weights = np.random.rand(self.nodes[0]+1, self.nodes[1])


    def _feedforward(self, X: np.ndarray):
        
        if len(X.shape) == 1: 
            X = X.reshape((1, X.shape[0]))

        # Adding bias to the data
        bias = np.ones(X.shape[0]) * 0.01
        X = np.hstack([bias, X])
        
        self.z_matrix = np.zeros((X.shape[0], self.weights.shape[0]))
        self.a_matrix = np.zeros((X.shape[0], self.weights.shape[0]))

        for i in range(self.weights.shape[0]): 
            z = X @ self.weights[i]
            self.z_matrix[i, :] = z[:]  
            a = self.act_func(z)
            self.a_matrix[i, : ] = a[:]

        return self.a_matrix


    def _backpropagate(self, delta_next, lam):
        
        activation_derivative = derivate(self.act_func)

        delta_matrix = (self.weights[1:, :] @ delta_next.T).T * activation_derivative(self.z_matrix)
        gradient_matrix = self.a_matrix[:, 1:].T @ delta_matrix
        gradient_bias = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])

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

        
# TODO: Test if OutputLayer inherits the constructor in the desired way, or if some changes need to be applied.        
class OutputLayer(FullyConnectedLayer):

    def __init__(
        self, 
        nodes: tuple[int], 
        output_func: Callable,
        cost_func: Callable,  
        scheduler: Scheduler,
        *scheduler_args: list, 
        seed=None,
    ): 
        super().__init__(nodes, output_func, scheduler, *scheduler_args, seed)
        
        self.cost_func = self.cost_func

        self.set_prediction() # Decides what type of prediction the output layer performs

        
    def _feedforward(self, X: np.ndarray):
        
        if len(X.shape) == 1: 
            X = X.reshape((1, X.shape[0]))

        # Adding bias to the data
        bias = np.ones(X.shape[0]) * 0.01
        X = np.hstack([bias, X])
        
        self.z_matrix = np.zeros((X.shape[0], self.weights.shape[0]))
        self.a_matrix = np.zeros((X.shape[0], self.weights.shape[0]))

        for i in range(self.weights.shape[0]): 
            try: 
                z = X @ self.weights[i]
                self.z_matrix[i, :] = z[:]  
                a = self.act_func(z) # Here the act_func is the output_func
                self.a_matrix[i, :] = a[:]
            except Exception as OverflowError: 
                print('Overflow encountered in fit(): Consider lowering your learning rate or scheduler specific parameters such as momentum, or check if your input values need scaling ')

        return self.a_matrix


    def _backpropagate(self, target, lam):
        
        # Again, remember that in hte OutputLayer the activation function is the output function
        activation_derivative = derivate(self.act_func) 

        if self.prediction is 'Multi-class': 
            delta_matrix = self.a_matrix - target
        else: 
            cost_func_derivative = grad(self.cost_func)
            delta_matrix = activation_derivative(self.z_matrix) * cost_func_derivative(self.a_matrix)
        
        gradient_matrix = self.a_matrix[:, 1:].T @ delta_matrix
        gradient_bias = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])

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


    def predict(self, X: np.ndarra , *, threshold=0.5): 
        
        predict = self._feedforward(X)

        if self.prediction is 'Binary': 
            return np.where(predict > threshold, 1, 0)
        else: 
            return predict

    
    def set_prediction(self): 

        if self.act_func.__name__ is None: 
            self.prediction = 'Regression'
        elif self.act_func.__name__ == 'sigmoid' or self.act_func.__name__ == 'tanh': 
            self.prediction = 'Binary'
        else: 
            self.prediction = 'Mulit-class'      



class Convolution2DLayer(Layer): 

    def __init__(
        self, 
        input_channels,
        feature_maps, # also known as feature maps
        kernel_size, 
        stride, 
        padding, 
        act_func, 
    ): 
        super().__init__()
        self.kernel_size = kernel_size 
        self.input_channels = input_channels
        self.feature_maps = feature_maps
        self.stride = stride 
        self.padding = padding
        self.act_func = act_func
        
    def _initialize_weights(self):

        if self.seed is not None: 
            np.random.seed(self.seed)

        kernel_tensor = np.ndarray(
                                    (
                                    self.input_channels, self.feature_maps, 
                                    self.kernel_size, self.kernel_size
                                    )
                                )

        for i in range(kernel_tensor.shape[0]): 
            for j in range(kernel_tensor.shape[1]): 
                kernel_tensor[i,j,:,:] = np.random.rand(self.kernel_size, self.kernel_size)
                
        return kernel_tensor

    def _reset_weights(self): 
        
        if self.seed is not None: 
            np.random.seed(self.seed)

        self.weights = np.random.rand(self.nodes[0]+1, self.nodes[1])

class Pooling2DLayer(Layer):

    def __init__(self):
        super().__init__()

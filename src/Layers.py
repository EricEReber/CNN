import math
import autograd.numpy as np
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

    def _update_weights(self): 
        raise NotImplementedError

class FullyConnectedLayer(Layer): 
    
    def __init__(
            self, 
            nodes: list[int], 
            act_func: Callable,  
            scheduler: Scheduler, 
            seed=None,
    ):

        self.nodes = nodes
        self.act_func = act_func
        self.scheduler = scheduler
        self.weights = self._initialize_weights()
        self.seed = seed

        self._initialize_weights()

    def _initialize_weights(self):

        if self.seed is not None: 
            np.random.seed(self.seed)

        weight_matrix = np.random.rand(self.nodes[0]+1, self.nodes[1])
        
        return weight_matrix

    def _reset_weights(self): 
        
        if self.seed is not None: 
            np.random.seed(self.seed)

        self.weights = np.random.rand(self.nodes[0]+1, self.nodes[1])
 
    def _update_weights(self): 
        raise NotImplementedError

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

        update_matrix = np.vstack( 
        )
        
        
         

class OutputLayer(FullyConnectedLayer):

    def __init__(self): 
        super().__init__()

class Convolution2DLayer(Layer): 

    def __init__(self): 
        super().__init__() 

class Pooling2DLayer(Layer):

    def __init__(self):
        super().__init__()

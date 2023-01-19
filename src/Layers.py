import math
import autograd.numpy as np
import sys
from typing import Callable 
from Schedulers import *

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
    
    def __init__(self, nodes, act_func: Callable, scheduler: Scheduler):
        self.nodes: list[int] = nodes
        self.act_func: Callable = act_func
        self.scheduler = scheduler
        self.weights: np.ndarray = self._initialize_weights()

    def _initialize_weights(self):
        
        self.weights = np.random.rand(self.nodes[0]+1, self.nodes[1])
    
    def _reset_weights(self): 
        
        self._initialize_weights()

    def _update_weights(self): 
        raise NotImplementedError

        
        
        

class OutputLayer(FullyConnectedLayer):

    def __init__(self): 
        super().__init__()

class Convolution2DLayer(Layer): 

    def __init__(self): 
        super().__init__() 

class Pooling2DLayer(Layer):

    def __init__(self):
        super().__init__()

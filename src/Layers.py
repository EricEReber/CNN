import math
import autograd.numpy as np
import sys

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
    
    def __init__(self): 
        super().__init__()

class OutputLayer(FullyConnectedLayer):

    def __init__(self): 
        super().__init__()

class Convolution2DLayer(Layer): 

    def __init__(self): 
        super().__init__() 

class Pooling2DLayer(Layer):

    def __init__(self):
        super().__init__()

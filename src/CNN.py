import math
import autograd.numpy as np
import sys
import warnings
from Schedulers import *
from activationFunctions import *
from costFunctions import *
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy
from typing import Tuple, Callable
from sklearn.utils import resample
from Layers import Layer

warnings.simplefilter("error")

class CNN:
 
    def __init__(
        self,
        layers: np.ndarray[Layer],
        cost_func: Callable = CostOLS,
        seed: int = None,
    ):
        self.layers = layers
        self.cost_func = cost_func
        self.seed = seed
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.a_matrices = list()
        self.z_matrices = list()
        self.classification = None

        self._initialize_weights()
        self._set_classification()

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        scheduler_class: Scheduler,
        *scheduler_args: list(),
        batches: int = 1,
        epochs: int = 100,
        lam: float = 0,
        X_val: np.ndarray = None,
        t_val: np.ndarray = None,
    ):
        #TODO: With the new code architecture, the fit method has to be updated in order
        # to take advantage of the modular design 

        raise NotImplementedError
        
    def _feedforward(self, X: np.ndarray):
        #TODO - Implement a version of feed forward that uses Layer-classes
        raise NotImplementedError

    def _backpropagate(self, X, t, lam):
        #TODO - Implement a version of backpropagation that uses Layer-classes
        raise NotImplementedError
    


    def _accuracy(self, prediction: np.ndarray, target: np.ndarray):
        """
        Description:
        ------------
            Calculates accuracy of given prediction to target

        Parameters:
        ------------
            I   prediction (np.ndarray): vector of predicitons output network
            (1s and 0s in case of classification, and real numbers in case of regression)
            II  target (np.ndarray): vector of true values (Ideally what the network should predict)

        Returns:
        ------------
            A floating point number representing the percentage of correctly classified instances.
        """
        assert prediction.size == target.size
        return np.average((target == prediction))

    def _initialize_weights(self):
        #TODO: Write code to initialize each layers weights
       raise NotImplementedError
       
    def _set_classification(self):
        self.classification = False
        if (
            self.cost_func.__name__ == "CostLogReg"
            or self.cost_func.__name__ == "CostCrossEntropy"
        ):
            self.classification = True

    def _progress_bar(self, progression, **kwargs):
        """
        Description:
        ------------
            Displays progress of training
        """
        print_length = 40
        num_equals = int(progression * print_length)
        num_not = print_length - num_equals
        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = self._fmt(progression * 100, N=5)
        line = f"  {bar} {perc_print}% "

        for key in kwargs:
            if kwargs[key]:
                value = self._fmt(kwargs[key], N=4)
                line += f"| {key}: {value} "
        print(line, end="\r")
        return len(line)

    def _fmt(self, value, N=4):
        """
        Description:
        ------------
            Formats decimal numbers for progress bar
        """
        if value > 0:
            v = value
        elif value < 0:
            v = -10 * value
        else:
            v = 1
        n = 1 + math.floor(math.log10(v))
        if n >= N - 1:
            return str(round(value))
            # or overflow
        return f"{value:.{N-n-1}f}"

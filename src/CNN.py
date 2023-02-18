import math
import autograd.numpy as np
import sys
import warnings
from src.Schedulers import *
from src.activationFunctions import *
from src.costFunctions import *
from src.Layers import *
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy
from typing import Tuple, Callable
from sklearn.utils import resample
from collections import OrderedDict

warnings.simplefilter("error")


class CNN:
    def __init__(
        self,
        cost_func: Callable = CostOLS,
        scheduler: Scheduler = Adam,
        seed: int = None,
    ):
        self.layers = list()
        self.cost_func = cost_func
        self.scheduler = scheduler
        self.seed = seed
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        # self.a_matrices = list()
        # self.z_matrices = list()
        self.classification = None

        self._set_classification()

    def add_FullyConnectedLayer(self, nodes, act_func, scheduler=None, seed=None):
        # TODO efficient way to replace final FullyConnectedLayer with Output
        # future idea: have this function (and similar functions) add
        # 'initialization of FullyConnectedLayer' to a queue. Layers in queue
        # are initialized when fit() is called, final layer in queue becomes
        # OutputLayer. Saves us from changing final layer every new layer.

        # assert self.layers, "FullyConnectedLayer should not be first added layer"

        if scheduler is None:
            scheduler = self.scheduler

        if self.layers:
            prev_nodes = self.layers[-1].nodes[1]
            layer = FullyConnectedLayer([prev_nodes, nodes], act_func, scheduler, seed)
        else:
            # only for testing, FullyConnectedLayer should always follow
            # FullyConnectedLayer or FlattenLayer
            layer = FullyConnectedLayer(nodes, act_func, scheduler, seed)
        self.layers.append(layer)

    def add_OutputLayer(self, nodes, output_func, scheduler=None, seed=None):
        assert self.layers, "OutputLayer should not be first added layer"

        if scheduler is None:
            scheduler = self.scheduler

        prev_nodes = self.layers[-1].nodes[1]
        output_layer = OutputLayer(
            [prev_nodes, nodes], output_func, self.cost_func, scheduler, seed
        )
        self.layers.append(output_layer)

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        # scheduler_class: Scheduler,
        batches: int = 1,
        epochs: int = 100,
        lam: float = 0,
        X_val: np.ndarray = None,
        t_val: np.ndarray = None,
    ):
        # for consecutive calls of fit()
        for layer in self.layers:
            layer._reset_weights()

        # setup 
        if self.seed is not None:
            np.random.seed(self.seed)

        val_set = False
        if X_val is not None and t_val is not None:
            val_set = True

        # creating arrays for score metrics
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)
        val_errors = np.empty(epochs)
        val_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        val_accs = np.empty(epochs)
        val_accs.fill(np.nan)

        batch_size = X.shape[0] // batches

        X, t = resample(X, t)

        cost_function_train = self.cost_func(t)
        if val_set:
            cost_function_val = self.cost_func(t_val)

        try:
            for epoch in range(epochs):
                for batch_num in range(batches):
                    if batch_num == batches - 1:
                        # If the for loop has reached the last batch_num, take all thats left
                        X_batch = X[batch_num * batch_size :, :]
                        t_batch = t[batch_num * batch_size :, :]
                    else:
                        X_batch = X[batch_num * batch_size : (batch_num + 1) * batch_size, :]
                        t_batch = t[batch_num * batch_size : (batch_num + 1) * batch_size, :]

                    self._feedforward(X_batch)
                    self._backpropagate(t_batch, lam)

                # reset schedulers for each epoch (some schedulers pass in this call)
                for layer in self.layers:
                    if isinstance(layer, FullyConnectedLayer):
                        layer.reset_scheduler()

                # computing performance metrics
                pred_train = self.predict(X)
                train_error = cost_function_train(pred_train)

                train_errors[e] = train_error
                if val_set:
                    
                    pred_val = self.predict(X_val)
                    val_error = cost_function_val(pred_val)
                    val_errors[e] = val_error

                if self.classification:
                    train_acc = self._accuracy(self.predict(X), t)
                    train_accs[e] = train_acc
                    if val_set:
                        val_acc = self._accuracy(pred_val, t_val)
                        val_accs[e] = val_acc

                # printing progress bar
                progression = e / epochs
                print_length = self._progress_bar(
                    progression,
                    train_error=train_errors[e],
                    train_acc=train_accs[e],
                    val_error=val_errors[e],
                    val_acc=val_accs[e],
                )
        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

        # visualization of training progression (similiar to tensorflow progression bar)
        sys.stdout.write("\r" + " " * print_length)
        sys.stdout.flush()
        self._progress_bar(
            1,
            train_error=train_errors[e],
            train_acc=train_accs[e],
            val_error=val_errors[e],
            val_acc=val_accs[e],
        )
        sys.stdout.write("")

        # return performance metrics for the entire run
        scores = dict()

        scores["train_errors"] = train_errors

        if val_set:
            scores["val_errors"] = val_errors

        if self.classification:
            scores["train_accs"] = train_accs

            if val_set:
                scores["val_accs"] = val_accs

        return scores

        # TODO add batches, val set

    def _feedforward(self, X: np.ndarray):

        a = X
        for layer in self.layers:
            a = layer._feedforward(a)

        return a

    def _backpropagate(self, t, lam):
        reversed_layers = self.layers[::-1]

        for layer in reversed_layers:
            if isinstance(layer, OutputLayer):
                weights_next, delta_next = layer._backpropagate(t, lam)
            elif isinstance(layer, FullyConnectedLayer):
                weights_next, delta_next = layer._backpropagate(
                    weights_next, delta_next, t, lam
                )
            else:
                # TODO implement other types of layers
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

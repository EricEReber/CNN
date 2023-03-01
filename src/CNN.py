import math
import autograd.numpy as np
import sys
import warnings
from src.Schedulers import *
from src.activationFunctions import *
from src.costFunctions import *
from src.ProgressBar import *
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
        cost_func: Callable = CostLogReg,
        scheduler: Scheduler = Adam(1e-4, 0.9, 0.999),
        seed: int = None,
    ):
        self.layers = list()
        self.cost_func = cost_func
        self.scheduler = scheduler
        self.seed = seed
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.prediction = None
        self.batches = 1

        self._set_classification()

    def add_FullyConnectedLayer(self, nodes, act_func, scheduler=None, seed=None):
        if scheduler is None:
            scheduler = self.scheduler

        layer = FullyConnectedLayer(nodes, act_func, scheduler, seed)
        self.layers.append(layer)

    def add_OutputLayer(self, nodes, output_func, scheduler=None, seed=None):
        assert self.layers, "OutputLayer should not be first added layer"

        if scheduler is None:
            scheduler = self.scheduler

        output_layer = OutputLayer(nodes, output_func, self.cost_func, scheduler, seed)
        self.layers.append(output_layer)
        self.prediction = output_layer.get_prediction()

    def add_FlattenLayer(self, seed=None):
        self.layers.append(FlattenLayer(seed))

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        epochs: int = 100,
        lam: float = 0,
        batches: int = 1,
        X_val: np.ndarray = None,
        t_val: np.ndarray = None,
    ):
        # for consecutive calls of fit()
        prev_nodes = X.shape[1] * X.shape[2] * X.shape[3]
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                prev_nodes = layer._reset_weights(prev_nodes)

        # setup
        if self.seed is not None:
            np.random.seed(self.seed)

        # creating arrays for score metrics
        scores = self._initialize_scores(epochs)

        try:
            batches = X.shape[0]

            for epoch in range(epochs):
                for batch in range(batches):
                    X_batch = X[batch, :, :, :]
                    self._feedforward(X_batch)
                    self._backpropagate(t, lam)

                # reset schedulers for each epoch (some schedulers pass in this call)
                for layer in self.layers:
                    if isinstance(layer, FullyConnectedLayer):
                        layer._reset_scheduler()

                # computing performance metrics
                scores = self._compute_scores(scores, epoch, X, t, X_val, t_val)

                # printing progress bar
                print_length = self._progress_bar(
                    epoch,
                    epochs,
                    scores,
                )
        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

        # visualization of training progression (similiar to tensorflow progression bar)
        sys.stdout.write("\r" + " " * print_length)
        sys.stdout.flush()
        self._progress_bar(
            epochs,
            epochs,
            scores,
        )
        sys.stdout.write("")

        return scores

    def _feedforward(self, X_batch):
        a = None
        for layer in self.layers:

            if isinstance(layer, FlattenLayer):
                a = layer._feedforward(X_batch)

            elif isinstance(layer, FullyConnectedLayer):
                assert a is not None

                a = layer._feedforward(a)

            else:
                # TODO implement other types of layers
                raise NotImplementedError

        return a

    def _backpropagate(self, t, lam):
        reversed_layers = self.layers[::-1]

        for i in range(len(reversed_layers) - 1):
            layer = reversed_layers[i]
            prev_layer = reversed_layers[i + 1]
            if isinstance(layer, OutputLayer):
                weights_next, delta_next = layer._backpropagate(
                    t, prev_layer.get_prev_a(), lam
                )
            elif isinstance(layer, FullyConnectedLayer):
                weights_next, delta_next = layer._backpropagate(
                    weights_next, delta_next, prev_layer.get_prev_a(), lam
                )
            elif isinstance(layer, FlattenLayer):
                delta_next = layer._backpropagate(delta_next)
            else:
                # TODO implement other types of layers
                raise NotImplementedError

    def _compute_scores(self, scores, epoch, X, t, X_val, t_val):

        pred_train = self.predict(X)
        cost_function_train = self.cost_func(t)
        train_error = cost_function_train(pred_train)
        scores["train_errors"][epoch] = train_error

        if X_val is not None and t_val is not None:
            cost_function_val = self.cost_func(t_val)
            pred_val = self.predict(X_val)
            val_error = cost_function_val(pred_val)
            scores["val_errors"][epoch] = val_error

        if self.prediction != "Regression":
            train_acc = self._accuracy(pred_train, t)
            scores["train_accs"][epoch] = train_acc
            if X_val is not None and t_val is not None:
                val_acc = self._accuracy(pred_val, t_val)
                scores["val_accs"][epoch] = val_acc

        return scores

    def _initialize_scores(self, epochs):
        scores = dict()

        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)
        val_errors = np.empty(epochs)
        val_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        val_accs = np.empty(epochs)
        val_accs.fill(np.nan)

        scores["train_errors"] = train_errors
        scores["val_errors"] = val_errors
        scores["train_accs"] = train_accs
        scores["val_accs"] = val_accs

        return scores

    def predict(self, X: np.ndarray, *, threshold=0.5):

        predict = np.zeros((X.shape[0], 1))
        for X_batch in range(X.shape[0]):
            predict[X_batch, :] = self._feedforward(X[X_batch, :, :, :])

        if self.prediction == "Binary":
            return np.where(predict > threshold, 1, 0)
        else:
            return predict

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

    def _progress_bar(self, epoch, epochs, scores):
        """
        Description:
        ------------
            Displays progress of training
        """
        progression = epoch / epochs
        epoch -= 1
        print_length = 40
        num_equals = int(progression * print_length)
        num_not = print_length - num_equals
        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = self._fmt(progression * 100, N=5)
        line = f"  {bar} {perc_print}% "

        for key, score in scores.items():
            if np.isnan(score[epoch]) == False:
                value = self._fmt(score[epoch], N=4)
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

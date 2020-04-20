import math
import numpy as np


def relu_function(x):
    return x if x > 0 else 0


def relu_derivative(x):
    return 1 if x > 0 else 0


def logistic_function(x):
    return 0 if x < -1e2 else 1. / (1 + math.exp(-x))


def logistic_derivative(x):
    return x * (1 - x)


class ActivationRectifiedLinearUnit:
    @staticmethod
    def apply_to_column(activation_input):
        return np.vectorize(relu_function)(activation_input)

    @staticmethod
    def gradient_wrt_activation_input(activation_output):
        derivative_column = np.vectorize(relu_derivative)(activation_output)
        return np.diag(derivative_column.flatten())


class ActivationLogistic:
    @staticmethod
    def apply_to_column(activation_input):
        return np.vectorize(logistic_function)(activation_input)

    @staticmethod
    def gradient_wrt_activation_input(activation_output):
        derivative_column = logistic_derivative(activation_output)
        return np.diag(derivative_column.flatten())


class ActivationSoftmax:
    @staticmethod
    def apply_to_column(activation_input):
        exp = np.vectorize(math.exp)(activation_input)
        return exp / sum(exp)

    @staticmethod
    def gradient_wrt_activation_input(activation_output):
        return np.diag(activation_output.flatten()) - activation_output * np.transpose(activation_output)


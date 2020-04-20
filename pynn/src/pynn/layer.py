import numpy as np


def compute_gradients(dC_dA, dA_dZ, dZ_dW, dZ_dX):
    dC_dZ = np.matmul(dC_dA, dA_dZ)

    dC_dW_transpose = np.matmul(dZ_dW, dC_dZ)
    dC_dW = np.transpose(dC_dW_transpose)

    dC_dX = np.matmul(dC_dZ, dZ_dX)

    return dC_dW, dC_dX


class Layer:
    def __init__(self, activation):
        self.weights = []  # random
        self.bias = []
        self.activation = activation
        self.recent_input = []
        self.recent_output = []
        self.weight_change = []
        self.bias_change = []

    def __weights_with_bias(self):
        # this could eventually be conditional on actually wanting bias
        return np.concatenate((self.weights, self.bias), axis=1)

    def __input_vector_with_unity(self, input_vector):
        # this could eventually be conditional on actually wanting bias
        return np.concatenate((input_vector, np.array([[1]])), axis=0)

    def run(self, input_vector):
        weights_with_bias = self.__weights_with_bias()
        input_vector_with_unity = self.__input_vector_with_unity(input_vector)

        activation_input = np.matmul(weights_with_bias, input_vector_with_unity)
        activation_output = self.activation.apply_to_column(activation_input)

        self.recent_input = input_vector_with_unity
        self.recent_output = activation_output
        return activation_output

    def calculate_update(self, dC_dA, learning_rate):
        dA_dZ = self.activation.gradient_wrt_activation_input(self.recent_output)
        dZ_dW = self.recent_input
        dZ_dX = self.__weights_with_bias()

        dC_dW, dC_dX = compute_gradients(dC_dA, dA_dZ, dZ_dW, dZ_dX)

        weights_with_bias_change = -learning_rate * dC_dW
        self.weight_change = weights_with_bias_change[:, :-1]
        self.bias_change = weights_with_bias_change[:, -1:]

        return dC_dX[:, :-1]

    def apply_update(self):
        self.weights += self.weight_change
        self.bias += self.bias_change

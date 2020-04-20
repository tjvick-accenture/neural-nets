import numpy as np


class Layer:
    def __init__(self, activation):
        self.weights = []  # random
        self.bias = []  # include as weights?
        self.activation = activation
        self.recent_input = []
        self.recent_output = []
        self.weight_change = []

    def run(self, input_vector):
        weights_with_bias = np.concatenate((self.weights, self.bias), axis=1)
        input_vector_with_constant = np.concatenate((input_vector, np.array([[1]])), axis=0)

        activation_input = np.matmul(weights_with_bias, input_vector_with_constant)
        activation_output = self.activation.apply_to_column(activation_input)

        # self.recent_input = input_vector
        # self.recent_output = activation_output
        return activation_output

    def calculate_update(self, dC_dA, learning_rate):
        dA_dZ = self.activation.gradient_wrt_activation_input(self.recent_output)
        dZ_dW = self.weights
        dC_dW = self.compute_gradient_dW(dC_dA, dA_dZ, dZ_dW)

        weight_change = -learning_rate * dC_dW
        self.weight_change = weight_change

        dZ_dA = self.weights
        return self.compute_gradient_dA(dC_dA, dA_dZ, dZ_dA)

    # static
    def compute_gradient_dW(self, dC_dA, dA_dZ, dZ_dW):
        return []

    # static
    def compute_gradient_dA(self, dC_dA, dA_dZ, dZ_dA):
        return []

    def apply_update(self):
        self.weights += self.weight_change
import numpy as np
from src.pynn.activation import *
from src.pynn.layer import *


class TestLayerRun:
    def test_zero_weights_and_bias_yield_zero_output(self):
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.weights = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        layer.bias = np.array([0, 0, 0]).reshape(3, 1)

        input_vector = np.array([1, 2, 3]).reshape(3, 1)

        output = layer.run(input_vector)

        expected = np.array([0, 0, 0]).reshape(3, 1)

        np.testing.assert_array_equal(output, expected)

    def test_applies_activation_function_to_product_of_inputs_and_weights_without_bias(self):
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.weights = np.array([
            [1, 1/2, 1/3],
            [2, 2/2, 2/3],
            [3, 3/2, 3/3]
        ])
        layer.bias = np.array([0, 0, 0]).reshape(3, 1)

        input_vector = np.array([1, 2, 3]).reshape(3, 1)

        output = layer.run(input_vector)

        expected = np.array([3, 6, 9]).reshape(3, 1)

        np.testing.assert_array_equal(output, expected)

    def test_applies_activation_function_to_product_of_inputs_and_weights_with_bias(self):
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.weights = np.array([
            [1, 1/2, 1/3],
            [2, 2/2, 2/3],
            [3, 3/2, 3/3]
        ])
        layer.bias = np.array([-3, -2, -1]).reshape(3, 1)

        input_vector = np.array([1, 2, 3]).reshape(3, 1)

        output = layer.run(input_vector)

        expected = np.array([0, 4, 8]).reshape(3, 1)

        np.testing.assert_array_equal(output, expected)


class TestLayerCalculateUpdate:

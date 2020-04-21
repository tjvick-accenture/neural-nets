import numpy as np
from src.pynn.activation import *
from src.pynn.layer import *
from test.utils import column


class TestLayerRun:
    def test_zero_weights_and_bias_yield_zero_output(self):
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.weights = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        layer.bias = column([0, 0, 0])

        input_vector = column([1, 2, 3])

        output = layer.run(input_vector)

        expected = column([0, 0, 0])

        np.testing.assert_array_equal(output, expected)

    def test_applies_activation_function_to_product_of_inputs_and_weights_with_zero_bias(self):
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.weights = np.array([
            [-1, -1/2, -1/3],
            [1, 1/2, 1/3],
            [2, 2/2, 2/3],
            [3, 3/2, 3/3]
        ])
        layer.bias = column([0, 0, 0, 0])

        input_vector = column([1, 2, 3])

        output = layer.run(input_vector)

        expected = column([0, 3, 6, 9])

        np.testing.assert_array_equal(output, expected)

    def test_applies_activation_function_to_product_of_inputs_and_weights_with_bias(self):
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.weights = np.array([
            [-1, -1/2, -1/3],
            [1, 1/2, 1/3],
            [2, 2/2, 2/3],
            [3, 3/2, 3/3]
        ])
        layer.bias = column([-4, -3, -2, -1])

        input_vector = column([1, 2, 3])

        output = layer.run(input_vector)

        expected = column([0, 0, 4, 8])

        np.testing.assert_array_equal(output, expected)


class TestLayerCalculateUpdate:
    def test_calculates_weight_change_using_gradient_wrt_weights(self):
        # ARRANGE
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.bias = column([0, 0, 0])
        layer.weights = np.array([
            [-1, -1, -1],
            [1, 1, 1],
            [2, 2, 2]
        ])

        # ACT
        input_vector = column([1, 2, 3])
        layer.run(input_vector)

        dC_dA = np.array([[0.25, 0.5, 1]])
        layer.calculate_update(dC_dA, 1)

        # output = relu(weights * input) = [0, 6, 12]
        # dA_dZ  = relu_gradient(output) = diag([0, 1, 1])
        # dZ_dW  = input = [1 2 3]
        # dC_dW  = T(dZ_dW * dC_dA * dA_dW)
        #        = T([1 2 3]T * [0.25 0.5 1] * diag([0 1 1]))

        # ASSERT
        expected_weight_change = -np.array([
            [0, 0, 0],
            [0.5, 1, 1.5],
            [1, 2, 3]
        ])

        np.testing.assert_array_equal(layer.weight_change, expected_weight_change)

    def test_calculates_bias_change_using_gradient_wrt_bias(self):
        # ARRANGE
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.bias = column([-1, 1, 2])
        layer.weights = np.array([
            [-1, -1],
            [1, 1],
            [2, 2]
        ])

        # ACT
        input_vector = column([2, 3])
        layer.run(input_vector)

        dC_dA = np.array([[0.25, 0.5, 1]])
        layer.calculate_update(dC_dA, 1)

        # ASSERT
        # Same case as above test, just with bias instead of the first weights column
        expected_bias_change = -column([0, 0.5, 1])

        np.testing.assert_array_equal(layer.bias_change, expected_bias_change)

    def test_calculates_gradient_wrt_inputs(self):
        # ARRANGE
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.bias = column([0, 0, 0])
        layer.weights = np.array([
            [-1, -1, -1],
            [1, 1.2, 1.5],
            [2, 2.5, 3]
        ])

        # ACT
        input_vector = column([1, 2, 3])
        layer.run(input_vector)

        dC_dA = np.array([[0.25, 0.5, 1]])
        gradient_wrt_input = layer.calculate_update(dC_dA, 1)

        # output = relu(weights * input) = [0, 6, 12]
        # dA_dZ  = relu_gradient(output) = diag([0, 1, 1])
        # dZ_dX  = weights
        # dC_dX  = dC_dA * dA_dZ * dZ_dX)
        #        = [0.25 0.5 1] * diag([0 1 1])) * weights
        #        = [0 0.5 1] * weights

        # ASSERT
        expected_gradient_wrt_input = np.array([
            [2.5, 3.1, 3.75]
        ])

        np.testing.assert_array_equal(gradient_wrt_input, expected_gradient_wrt_input)
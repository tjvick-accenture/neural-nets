import numpy as np
from src.pynn.activation import *


class TestReluFunction:
    def test_returns_input_when_input_is_positive(self):
        assert relu_function(0.7) == 0.7
        assert relu_function(1.7) == 1.7
        assert relu_function(7.7) == 7.7

    def test_returns_zero_when_input_is_not_positive(self):
        assert relu_function(0) == 0
        assert relu_function(-0.7) == 0
        assert relu_function(-700) == 0


class TestLogisticFunction:
    def test_returns_half_when_input_is_0(self):
        assert logistic_function(0) == 0.5

    def test_increases_monotonically_towards_1_when_input_is_positive(self):
        assert logistic_function(1) > logistic_function(0)
        assert logistic_function(2) > logistic_function(1)
        assert logistic_function(10) > logistic_function(2)
        assert 1 > logistic_function(10)

    def test_increases_monotonically_from_0_when_input_is_negative(self):
        assert logistic_function(-10) > 0
        assert logistic_function(-2) > logistic_function(-10)
        assert logistic_function(-1) > logistic_function(-2)
        assert logistic_function(0) > logistic_function(-1)

    def test_returns_one_for_large_positive_values(self):
        assert logistic_function(1e3) == 1.
        assert logistic_function(1e16) == 1.

    def test_returns_zero_for_large_negative_values(self):
        assert logistic_function(-1e3) == 0
        assert logistic_function(-1e16) == 0


class TestActivationRectifiedLinearUnit:
    def test_applies_relu_function_to_column(self):
        column = np.array([-1, 0, 1, 2]).reshape(4, 1)
        expected = np.array([0, 0, 1, 2]).reshape(4, 1)

        activation = ActivationRectifiedLinearUnit()
        result = activation.apply_to_column(column)

        np.testing.assert_array_equal(result, expected)

    def test_calculates_gradient_of_relu_function(self):
        activation_output = np.array([-1, 0, 1, 2]).reshape(4, 1)

        activation = ActivationRectifiedLinearUnit()
        gradient = activation.gradient_wrt_activation_input(activation_output)

        expected = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        np.testing.assert_array_equal(gradient, expected)


class TestActivationLogistic:
    def test_applies_logistic_function_to_column(self):
        column_list = [-1, 0, 1, 2]
        expected_list = [logistic_function(x) for x in column_list]

        column = np.array(column_list).reshape(4, 1)
        expected = np.array(expected_list).reshape(4, 1)

        activation = ActivationLogistic()
        result = activation.apply_to_column(column)

        np.testing.assert_array_equal(result, expected)

    def test_calculates_gradient_of_logistic_function(self):
        activation_output = np.array([0.1, 0.3, 0.5, 1]).reshape(4, 1)

        activation = ActivationLogistic()
        gradient = activation.gradient_wrt_activation_input(activation_output)

        expected = np.array([
            [0.09, 0, 0, 0],
            [0, 0.21, 0, 0],
            [0, 0, 0.25, 0],
            [0, 0, 0, 0]
        ])

        np.testing.assert_array_almost_equal(gradient, expected, 15)


class TestActivationSoftmax:
    def test_applies_softmax_function_to_column(self):
        column_list = [-1, 0, 1, 2]
        exp_sum = sum(math.exp(x) for x in column_list)
        expected_list = [math.exp(x)/exp_sum for x in column_list]

        column = np.array(column_list).reshape(4, 1)
        expected = np.array(expected_list).reshape(4, 1)

        activation = ActivationSoftmax()
        result = activation.apply_to_column(column)

        np.testing.assert_array_equal(result, expected)

    def test_calculates_gradient_of_softmax_function(self):
        [a, b, c, d] = [0.1, 0.3, 0.5, 1]

        column = np.array([a, b, c, d]).reshape(4, 1)
        expected = np.array([
            [a*(1-a), -a*b, -a*c, -a*d],
            [-b*a, b*(1-b), -b*c, -b*d],
            [-c*a, -c*b, c*(1-c), -c*d],
            [-d*a, -d*b, -d*c, d*(1-d)]
        ])

        activation = ActivationSoftmax()
        result = activation.gradient_wrt_activation_input(column)

        np.testing.assert_array_almost_equal(result, expected, 15)

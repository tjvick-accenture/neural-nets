from src.pynn.loss import *
from test.utils import column


class TestLossAbsoluteError:
    def test_calculation_of_absolute_error_loss(self):
        loss_function = LossAbsoluteError

        output_vector = column([0, 1, 3])
        target_vector = column([1, 1, 1])

        result = loss_function.evaluate_loss(output_vector, target_vector)

        expected = 1

        assert result == expected

    def test_calculation_of_gradient_wrt_output(self):
        loss_function = LossAbsoluteError()

        output_vector = column([0, 1, 2])
        target_vector = column([1, 1, 1])
        result = loss_function.gradient_wrt_output(output_vector, target_vector)

        # sign(o_i - t_i)
        expected = np.array([[-1, 0, 1]])

        np.testing.assert_array_equal(result, expected)


class TestLossSquaredError:
    def test_calculation_of_squared_error_loss(self):
        loss_function = LossSquaredError

        output_vector = column([0, 1, 4])
        target_vector = column([1, 1, 1])

        result = loss_function.evaluate_loss(output_vector, target_vector)

        expected = 5

        assert result == expected

    def test_calculation_of_gradient_wrt_output(self):
        loss_function = LossSquaredError()

        output_vector = column([0, 1, 3])
        target_vector = column([1, 1, 1])
        result = loss_function.gradient_wrt_output(output_vector, target_vector)

        # o_i - t_i
        expected = np.array([[-1, 0, 2]])

        np.testing.assert_array_equal(result, expected)


class TestLossCategoricalCrossEntropy:
    def test_calculation_of_categorical_cross_entropy_loss(self):
        loss_function = LossCategoricalCrossEntropy

        def run_assertion(a, b, expected):
            output_vector = column(a)
            target_vector = column(b)
            result = loss_function.evaluate_loss(output_vector, target_vector)
            assert result == expected

        run_assertion([0.2, 0.3, 0.9], [0, 0, 1], -np.log(0.9))
        run_assertion([0.2, 0.3, 0.9], [0, 1, 0], -np.log(0.3))
        run_assertion([0.2, 0.3, 0.9], [0, 1, 1], -np.log(0.3)-np.log(0.9))
        run_assertion([0.2, 0.3, 1.0], [0, 1, 1], -np.log(0.3))
        run_assertion([0.2, 0.3, 0], [0, 0, 1], 1e100)
        run_assertion([0.2, 0.3, 0], [0, 0, -1], -1e100)
        run_assertion([0.2, 0.3, 0], [0, 0, 0], 0)

    def test_calculation_of_gradient_wrt_output(self):
        loss_function = LossCategoricalCrossEntropy()

        output_vector = column([0, 0, 1, 3, 4])
        target_vector = column([0, 1, 1, 1, 0])
        result = loss_function.gradient_wrt_output(output_vector, target_vector)

        # -t_i / o_i
        # division by zero cases:
        # target  = 0, output = 0  -> derivative = 0
        # target != 0, output = 0  -> derivative = 1e100 (might want something else eventually)
        expected = np.array([[0, -1e100, -1, -1 / 3, 0]])

        np.testing.assert_array_equal(result, expected)
import numpy as np

from pynn.activation import ActivationRectifiedLinearUnit
from pynn.layer import Layer
from pynn.loss import LossAbsoluteError
from pynn.network import Network
from pynn.trainer import Trainer
from test.utils import column


class TestTrainerReducesCost:
    def test_single_step_backpropagation_using_single_input_target_pair(self):
        # ARRANGE
        activation = ActivationRectifiedLinearUnit
        loss = LossAbsoluteError
        network, trainer = create_test_network_trainer(activation, loss)

        input_vector = column([0.2, 0.4, 0.8])
        target_vector = column([1, 0, 0])
        output_vector_i = network.run(input_vector)
        cost_i = loss.evaluate_loss(output_vector_i, target_vector)

        # ACT
        trainer.train([input_vector], [target_vector], 0.01, 1)

        # ASSERT
        output_vector_f = network.run(input_vector)
        cost_f = loss.evaluate_loss(output_vector_f, target_vector)

        assert cost_f < cost_i

    def test_single_epoch_backpropagation_using_specific_set_of_input_target_pairs(self):
        # ARRANGE
        activation = ActivationRectifiedLinearUnit
        loss = LossAbsoluteError
        network, trainer = create_test_network_trainer(activation, loss)

        input_vectors = [
            column([0.2, 0.4, 0.8]),
            column([0.1, 0.3, 0.7]),
            column([3, 2, 1])
        ]

        target_vectors = [
            column([1, 0, 0]),
            column([0, 1, 0]),
            column([0, 0, 1])
        ]

        epoch_cost_i = compute_epoch_cost(input_vectors, target_vectors, network, loss)
        trainer.train(input_vectors, target_vectors, 0.01, 1)
        epoch_cost_f = compute_epoch_cost(input_vectors, target_vectors, network, loss)

        assert epoch_cost_f < epoch_cost_i

    def test_single_epoch_backpropagation_using_random_set_of_input_target_pairs(self):
        # ARRANGE
        activation = ActivationRectifiedLinearUnit
        loss = LossAbsoluteError
        network, trainer = create_test_network_trainer(activation, loss)

        n_samples = 10
        input_vectors = [np.random.rand(3, 1) for _ in range(n_samples)]
        target_vectors = [np.random.rand(3, 1) for _ in range(n_samples)]

        epoch_cost_i = compute_epoch_cost(input_vectors, target_vectors, network, loss)
        trainer.train(input_vectors, target_vectors, 0.01, 1)
        epoch_cost_f = compute_epoch_cost(input_vectors, target_vectors, network, loss)

        assert epoch_cost_f < epoch_cost_i

    def test_multiple_epoch_backpropagation_using_random_set_of_input_target_pairs(self):
        # ARRANGE
        activation = ActivationRectifiedLinearUnit
        loss = LossAbsoluteError
        network, trainer = create_test_network_trainer(activation, loss)

        n_pairs = 3
        input_vectors = [np.random.rand(3, 1) for _ in range(n_pairs)]
        target_vectors = [np.random.rand(3, 1) for _ in range(n_pairs)]

        epoch_cost_0 = compute_epoch_cost(input_vectors, target_vectors, network, loss)
        trainer.train(input_vectors, target_vectors, 0.01, 10)
        epoch_cost_10 = compute_epoch_cost(input_vectors, target_vectors, network, loss)
        trainer.train(input_vectors, target_vectors, 0.01, 90)
        epoch_cost_100 = compute_epoch_cost(input_vectors, target_vectors, network, loss)

        assert epoch_cost_10 < epoch_cost_0
        assert epoch_cost_100 < epoch_cost_10


# TODO consider moving this (and similar tests) out into a separate file -
#  maybe a longer-running integration test group
class TestTrainerConvergesToKnownSolution:
    def test_network_weights_and_bias_converge_to_solution(self):
        weights, bias, input_vectors, target_vectors = generate_training_data(10)

        activation = ActivationRectifiedLinearUnit
        loss = LossAbsoluteError
        network, trainer = create_test_network_trainer(activation, loss)

        x = 0
        converged = False
        while not converged and x < 10000:
            trainer.train(input_vectors, target_vectors, 0.001, 1)
            weight_diff = difference_norm(network.layers[0].weights, weights)
            bias_diff = difference_norm(network.layers[0].bias, bias)
            x += 1
            if max(weight_diff, bias_diff) < 1e-2:
                converged = True

        assert converged


class TestTrainerReturnsCost:
    def test_cost_is_computed_for_each_epoch_and_returned(self):
        activation = ActivationRectifiedLinearUnit
        loss = LossAbsoluteError
        network, trainer = create_test_network_trainer(activation, loss)

        n_pairs = 5
        input_vectors = [np.random.rand(3, 1) for _ in range(n_pairs)]
        target_vectors = [np.random.rand(3, 1) for _ in range(n_pairs)]

        result_after_1 = trainer.train(input_vectors, target_vectors, 0.01, 1, cost_frequency=1)
        epoch_cost_1 = compute_epoch_cost(input_vectors, target_vectors, network, loss)
        result_after_2 = trainer.train(input_vectors, target_vectors, 0.01, 1, cost_frequency=1)
        epoch_cost_2 = compute_epoch_cost(input_vectors, target_vectors, network, loss)

        assert result_after_1 == [epoch_cost_1]
        assert result_after_2 == [epoch_cost_2]

    def test_cost_is_computed_after_every_nth_epoch_and_returned(self):
        activation = ActivationRectifiedLinearUnit
        loss = LossAbsoluteError
        network, trainer = create_test_network_trainer(activation, loss)

        n_pairs = 5
        input_vectors = [np.random.rand(3, 1) for _ in range(n_pairs)]
        target_vectors = [np.random.rand(3, 1) for _ in range(n_pairs)]

        result_after_10 = trainer.train(input_vectors, target_vectors, 0.01, 10, cost_frequency=10)
        epoch_cost_10 = compute_epoch_cost(input_vectors, target_vectors, network, loss)
        result_after_30 = trainer.train(input_vectors, target_vectors, 0.01, 20, cost_frequency=10)
        epoch_cost_30 = compute_epoch_cost(input_vectors, target_vectors, network, loss)

        assert result_after_10 == [epoch_cost_10]
        assert len(result_after_30) == 2
        assert result_after_30[-1] == [epoch_cost_30]

    def test_cost_is_not_computed_when_cost_frequency_not_defined(self):
        activation = ActivationRectifiedLinearUnit
        loss = LossAbsoluteError
        network, trainer = create_test_network_trainer(activation, loss)

        n_pairs = 5
        input_vectors = [np.random.rand(3, 1) for _ in range(n_pairs)]
        target_vectors = [np.random.rand(3, 1) for _ in range(n_pairs)]

        result_after_10 = trainer.train(input_vectors, target_vectors, 0.01, 10)

        assert result_after_10 is None


def create_test_network_trainer(activation, loss):
    layer = Layer(activation)
    layer.weights = np.random.rand(3, 3)
    layer.bias = np.random.rand(3, 1)
    network = Network()
    network.add(layer)

    trainer = Trainer(network, loss)
    return network, trainer


def compute_epoch_cost(input_vectors, target_vectors, network, loss):
    epoch_cost = 0
    for input_vector, target_vector in zip(input_vectors, target_vectors):
        output_vector = network.run(input_vector)
        cost = loss.evaluate_loss(output_vector, target_vector)
        epoch_cost += cost
    return epoch_cost


def generate_training_data(n_samples):
    weights = np.random.rand(3, 3)
    bias = np.random.rand(3, 1)
    input_vectors = [np.random.rand(3, 1) for _ in range(n_samples)]
    target_vectors = [np.matmul(weights, x) + bias for x in input_vectors]
    return weights, bias, input_vectors, target_vectors


def difference_norm(actual_array, target_array):
    return np.linalg.norm(target_array - actual_array, 1)
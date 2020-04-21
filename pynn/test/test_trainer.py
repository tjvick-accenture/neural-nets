import numpy as np

from pynn.activation import ActivationRectifiedLinearUnit, ActivationSoftmax
from pynn.layer import Layer
from pynn.loss import LossCategoricalCrossEntropy
from pynn.network import Network
from pynn.trainer import Trainer
from test.utils import column


class TestTrainer:
    def test_single_backpropagation_step_for_single_input_target_pair(self):
        # ARRANGE
        activation = ActivationRectifiedLinearUnit
        loss = LossCategoricalCrossEntropy
        network, trainer = create_test_network_trainer(activation, loss)

        input_vector = column([0.2, 0.4, 0.8])
        target_vector = column([1, 0, 0])
        output_vector_i = network.run(input_vector)
        cost_i = loss.evaluate_loss(output_vector_i, target_vector)

        # ACT
        trainer.train([input_vector], [target_vector], 0.001)

        # ASSERT
        output_vector_f = network.run(input_vector)
        cost_f = loss.evaluate_loss(output_vector_f, target_vector)

        assert cost_f < cost_i

    def test_backpropagation_for_specific_set_of_input_target_pairs(self):
        # ARRANGE
        activation = ActivationSoftmax
        loss = LossCategoricalCrossEntropy
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
        trainer.train(input_vectors, target_vectors, 0.1)
        epoch_cost_f = compute_epoch_cost(input_vectors, target_vectors, network, loss)

        assert epoch_cost_f < epoch_cost_i

    def test_backpropagation_for_random_set_of_input_target_pairs(self):
        # ARRANGE
        activation = ActivationSoftmax
        loss = LossCategoricalCrossEntropy
        network, trainer = create_test_network_trainer(activation, loss)

        n_pairs = 10
        input_vectors = [np.random.rand(3, 1) for _ in range(n_pairs)]
        target_vectors = [np.random.rand(3, 1) for _ in range(n_pairs)]

        epoch_cost_i = compute_epoch_cost(input_vectors, target_vectors, network, loss)
        trainer.train(input_vectors, target_vectors, 0.1)
        epoch_cost_f = compute_epoch_cost(input_vectors, target_vectors, network, loss)

        assert epoch_cost_f < epoch_cost_i


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

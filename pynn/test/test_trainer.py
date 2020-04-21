import numpy as np

from pynn.activation import ActivationRectifiedLinearUnit
from pynn.layer import Layer
from pynn.loss import LossCategoricalCrossEntropy
from pynn.network import Network
from pynn.trainer import Trainer
from test.utils import column


class TestTrainer:
    def test_single_backprop_step_for_single_input_target_pair(self):
        # ARRANGE
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.weights = np.random.rand(3, 3)
        layer.bias = np.random.rand(3, 1)
        network = Network()
        network.add(layer)

        loss = LossCategoricalCrossEntropy
        trainer = Trainer(network, loss)

        input_vector = column([0.2, 0.4, 0.8])
        target_vector = column([1, 0, 0])
        output_vector_0 = network.run(input_vector)
        cost_0 = loss.evaluate_loss(output_vector_0, target_vector)

        # ACT
        trainer.train([input_vector], [target_vector], 0.001)

        # ASSERT
        output_vector_1 = network.run(input_vector)
        cost_1 = loss.evaluate_loss(output_vector_1, target_vector)

        assert cost_1 < cost_0

import numpy as np
from pynn.activation import ActivationRectifiedLinearUnit
from pynn.layer import Layer
from pynn.loss import LossCategoricalCrossEntropy
from pynn.network import Network
from test.utils import column


class TestNetworkBackpropagation:
    def test_single_backprop_step_for_single_input_target_pair(self):
        # ARRANGE
        layer = Layer(ActivationRectifiedLinearUnit())
        layer.weights = np.random.rand(3, 3)
        layer.bias = np.random.rand(3, 1)
        network = Network()
        network.add(layer)

        # ACT
        input_vector = column([0.2, 0.4, 0.8])
        target_vector = column([1, 0, 0])
        output_vector = network.run(input_vector)

        loss = LossCategoricalCrossEntropy
        cost = loss.evaluate_loss(output_vector, target_vector)
        dC_dA = loss.gradient_wrt_output(output_vector, target_vector)

        network.calculate_update(dC_dA, 0.001)
        network.apply_update()

        # ASSERT
        new_output_vector = network.run(input_vector)
        new_cost = loss.evaluate_loss(new_output_vector, target_vector)

        assert new_cost < cost

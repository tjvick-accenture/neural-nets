import numpy as np
from pynn.activation import ActivationRectifiedLinearUnit
from pynn.layer import Layer
from pynn.network import Network
from test.utils import column


# class TestNetworkBackpropagation:
#     def test_single_backprop_step_for_single_input_target_pair(self):
#         network = Network()
#
#         layer = Layer(ActivationRectifiedLinearUnit())
#         layer.weights = np.array([[]])
#         layer.bias = column([0, 0, 0])
#
#         network.add(layer)
#
#         input_vector = column([])
#         target_vector = column([])
#
#         output_vector = network.run(input_vector)
#         cost = loss.evaluate(output_vector, target_vector)
#         dC_dA = loss.gradient_wrt_output(output_vector, target_vector)
#         layer.calculate_update(dC_dA, learning_rate)
#         layer.apply_update()
#
#         new_output_vector = network.run(input_vector)
#         new_cost = loss.evaluate(output_vector, target_vector)
#
#         assert new_cost < cost

class Trainer:
    def __init__(self, network, loss):
        self.network = network
        self.loss = loss

    def train(self, input_vectors, target_vectors, learning_rate):
        input_vector = input_vectors[0]
        target_vector = target_vectors[0]

        output_vector = self.network.run(input_vector)
        loss_gradient_wrt_output = self.loss.gradient_wrt_output(output_vector, target_vector)
        self.network.calculate_update(loss_gradient_wrt_output, learning_rate)
        self.network.apply_update()

# def train(input_vectors, target_vectors, n_epochs, loss, learning_rate):
#     network = Network()
#     layer = Layer()
#     network.add(layer)
#
#     for i_epoch in range(n_epochs):
#         # shuffle input_vectors?
#         for input_vector, target_vector in zip(input_vectors, target_vectors):
#             network.calculate_update(input_vector, target_vector, loss, learning_rate)
#             # could batch updates somehow
#             network.apply_update()


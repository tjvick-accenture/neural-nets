class Trainer:
    def __init__(self, network, loss):
        self.network = network
        self.loss = loss

    def train(self, input_vectors, target_vectors, learning_rate, n_epochs):
        # shuffle input vectors?
        for ix in range(n_epochs):
            for input_vector, target_vector in zip(input_vectors, target_vectors):
                output_vector = self.network.run(input_vector)
                loss_gradient_wrt_output = self.loss.gradient_wrt_output(output_vector, target_vector)
                self.network.calculate_update(loss_gradient_wrt_output, learning_rate)
                # could batch updates somehow
                self.network.apply_update()


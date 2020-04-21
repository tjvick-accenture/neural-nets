class Trainer:
    def __init__(self, network, loss):
        self.network = network
        self.loss = loss

    def train(self, input_vectors, target_vectors, learning_rate, n_epochs, cost_frequency=None):
        epoch_cost_history = [] if cost_frequency is not None else None

        for ix in range(n_epochs):
            for input_vector, target_vector in zip(input_vectors, target_vectors):
                output_vector = self.network.run(input_vector)
                loss_gradient_wrt_output = self.loss.gradient_wrt_output(output_vector, target_vector)
                self.network.calculate_update(loss_gradient_wrt_output, learning_rate)
                self.network.apply_update()

            if cost_frequency is not None:
                if (ix+1) % cost_frequency == 0:
                    epoch_cost_history.append(self.compute_epoch_cost(input_vectors, target_vectors))

        return epoch_cost_history

    def compute_epoch_cost(self, input_vectors, target_vectors):
        epoch_cost = 0
        for input_vector, target_vector in zip(input_vectors, target_vectors):
            output_vector = self.network.run(input_vector)
            cost = self.loss.evaluate_loss(output_vector, target_vector)
            epoch_cost += cost
        return epoch_cost

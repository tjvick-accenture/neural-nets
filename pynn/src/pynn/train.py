def train(input_vectors, target_vectors, n_epochs, loss, learning_rate):
    network = Network()
    layer = Layer()
    network.add(layer)

    for i_epoch in range(n_epochs):
        # shuffle input_vectors?
        for input_vector, target_vector in zip(input_vectors, target_vectors):
            network.calculate_update(input_vector, target_vector, loss, learning_rate)
            # could batch updates somehow
            network.apply_update()


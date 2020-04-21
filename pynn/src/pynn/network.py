class Network:
    def __init__(self):
        self.layers = []
        self.recent_output = None

    def add(self, layer):
        self.layers.append(layer)

    def run(self, input_vector):
        x = input_vector
        for layer in self.layers:
            x = layer.run(x)

        return x

    def calculate_update(self, dC_dA, learning_rate):
        for layer in reversed(self.layers):
            dC_dA = layer.calculate_update(dC_dA, learning_rate)

    def apply_update(self):
        for layer in self.layers:
            layer.apply_update()

    # Should a network care what loss function is used to train it?
    # NetworkTrainer(network, inputs, targets, loss)
    # NetworkTester(metwork, inputs, targets)
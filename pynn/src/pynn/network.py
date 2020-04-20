class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def run(self, input_vectors):
        inputs = input_vectors
        for layer in self.layers:
            inputs = layer.run(inputs)

        return inputs

    def calculate_update(self, input_vector, target_vector, loss, learning_rate):
        output_vector = self.run(input_vector)

        dC_dA = loss.gradient_wrt_output(output_vector, target_vector)
        for layer in reversed(self.layers):
            dC_dA = layer.calculate_update(dC_dA, learning_rate)

    def apply_update(self):
        for layer in self.layers:
            layer.update()
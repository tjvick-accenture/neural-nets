import numpy as np


class LossAbsoluteError:
    @staticmethod
    def evaluate_loss(output_vector, target_vector):
        return sum(output_vector - target_vector)

    @staticmethod
    def gradient_wrt_output(output_vector, target_vector):
        return np.sign(output_vector - target_vector)


class LossSquaredError:
    @staticmethod
    def gradient_wrt_output(output_vector, target_vector):
        return output_vector - target_vector


class LossCategoricalCrossEntropy:
    @staticmethod
    def gradient_wrt_output(output_vector, target_vector):
        return np.nan_to_num(-np.divide(target_vector, output_vector), neginf=-1e100)

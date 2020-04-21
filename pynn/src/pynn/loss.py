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
    def evaluate_loss(output_vector, target_vector):
        return 0.5 * sum((output_vector - target_vector) ** 2)

    @staticmethod
    def gradient_wrt_output(output_vector, target_vector):
        return output_vector - target_vector


class LossCategoricalCrossEntropy:
    @staticmethod
    def evaluate_loss(output_vector, target_vector):
        return np.nan_to_num(-sum(target_vector * np.log(output_vector)), posinf=1e100, neginf=-1e100)

    @staticmethod
    def gradient_wrt_output(output_vector, target_vector):
        return np.nan_to_num(-np.divide(target_vector, output_vector), neginf=-1e100)

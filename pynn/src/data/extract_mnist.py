import struct
from array import array
import numpy as np

np.set_printoptions(linewidth=260)


def extract_input_vectors_from_file(filename, n_samples=None):
    with open(filename, "rb") as f:
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", f.read(16))

        if n_samples is None:
            n_samples = n_images

        image_data = array("B", f.read(n_rows*n_cols*n_samples))

        images = []
        for ii in range(n_samples):
            image = np.asarray(image_data[ii*n_rows*n_cols:(ii+1)*n_rows*n_cols])
            images.append(image / 255)

    return images


def extract_target_vectors_from_file(filename, n_samples=None):
    with open(filename, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))

        if n_samples is None:
            n_samples = n_labels

        label_data = array("B", f.read(n_samples))
        labels = list(map(lambda x: one_hot_encode(x, 10), label_data))

    return labels


def one_hot_encode(value, length):
    x = np.zeros(length)
    x[value] = 1
    return x


# input_vectors = extract_input_vectors_from_file("data/external/mnist/train-images-idx3-ubyte", 1)
# target_vectors = extract_target_vectors_from_file("data/external/mnist/train-labels-idx1-ubyte", 1)

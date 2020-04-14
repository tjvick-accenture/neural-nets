require ExtractMNIST
require Logger
require ENN

training_images_filename = "data/raw/mnist/train-images-idx3-ubyte"
training_labels_filename = "data/raw/mnist/train-labels-idx1-ubyte"
testing_images_filename = "data/raw/mnist/t10k-images-idx3-ubyte"
testing_labels_filename = "data/raw/mnist/t10k-labels-idx1-ubyte"

n_samples = 1000
Logger.debug("Extracting training input vectors...")
training_input_vectors = ExtractMNIST.extract_input_vectors_from_file(training_images_filename, n_samples)
Logger.debug("Extracting training target vectors...")
training_target_vectors = ExtractMNIST.extract_target_vectors_from_file(training_labels_filename, n_samples)
Logger.debug("Extracting testing input vectors...")
testing_input_vectors = ExtractMNIST.extract_input_vectors_from_file(testing_images_filename, n_samples)
Logger.debug("Extracting testing target vectors...")
testing_target_vectors = ExtractMNIST.extract_target_vectors_from_file(testing_labels_filename, n_samples)


training_sequence = [0, 1, 5, 10]
for n_epochs <- training_sequence do
  Logger.debug("Training...")
  # in: network parameters
  # out: network (weights right now)
  weights = ENN.train(training_input_vectors, training_target_vectors, n_epochs, :softmax, :categorical_crossentropy)

  Logger.debug("Testing...")
  # in: network (weights right now)
  # out: performance
  results = ENN.test(testing_input_vectors, testing_target_vectors, weights, :softmax)

  Logger.debug("After #{n_epochs} training epochs")
  Logger.debug(inspect(results))
end
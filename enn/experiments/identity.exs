defmodule Identity do
  require Logger

  training_input_vectors = [
    Matrix.column([1.0, 0.0, 0.0]),
    Matrix.column([0.0, 1.0, 0.0]),
    Matrix.column([0.0, 0.0, 1.0]),
  ]
  training_target_vectors = [
    Matrix.column([1.0, 0.0, 0.0]),
    Matrix.column([0.0, 1.0, 0.0]),
    Matrix.column([0.0, 0.0, 1.0]),
  ]

  testing_input_vectors = [
    Matrix.column([0.6, 0.1, 0.1]),
    Matrix.column([0.1, 0.6, 0.1]),
    Matrix.column([0.1, 0.1, 0.6]),
  ]
  testing_target_vectors = [
    Matrix.column([1.0, 0.0, 0.0]),
    Matrix.column([0.0, 1.0, 0.0]),
    Matrix.column([0.0, 0.0, 1.0]),
  ]

  training_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]

  for n_cycles <- training_sequence do
    weights = ENN.train(training_input_vectors, training_target_vectors, n_cycles)
    results = ENN.test(testing_input_vectors, testing_target_vectors, weights)
    Logger.debug("After #{n_cycles} training cycles")
    Logger.debug(inspect(weights))
    Logger.debug(inspect(results))
  end
end
defmodule ENN do
  require Logger

  def apply_activation_function(x, activation_function \\ &Activation.step_function/1) do
    for [z] <- x do
      activation_function.(z)
    end
  end

  def perceptron_output(weights, bias, input) do
    Matrix.multiply(weights, input)
    |> Matrix.add(bias)
    |> apply_activation_function
  end

  def neuron_layer_output(input, weights, bias) do
    Matrix.multiply(weights, input)
    |> Matrix.add(bias)
    |> apply_activation_function(&Activation.logistic_function/1)
  end
end

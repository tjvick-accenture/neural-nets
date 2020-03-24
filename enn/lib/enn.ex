defmodule ENN do
  require Logger

  def step_function(x) when x >= 0 do
    1.0
  end

  def step_function(_x) do
    0.0
  end

  def apply_activation_function([x], activation_function \\ &step_function/1) do
    Enum.map(x, &activation_function.(&1))
  end

  def perceptron_output(weights, bias, input) do
    Matrix.multiply(weights, input)
    |> Matrix.add(bias)
    |> apply_activation_function
  end
end

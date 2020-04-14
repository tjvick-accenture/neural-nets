defmodule Activation do
  def function(a) do
    case a do
      :step -> &step_function/1
      :logistic -> &logistic_function/1
      :relu -> &relu_function/1
    end
  end

  def step_function(x) when x >= 0 do
    1.0
  end

  def step_function(_x) do
    0.0
  end

  def logistic_function(x) when 40 < x do
    1.0
  end

  def logistic_function(x) when -40 > x do
    0.0
  end

  def logistic_function(x) do
    1 / (1 + :math.exp(-x))
  end

  def relu_function(x) when x <= 0 do
    0.0
  end

  def relu_function(x) do
    x
  end

  def derivative_from_output(a) do
    case a do
      :logistic -> &logistic_derivative_from_output/1
      :relu -> &relu_derivative_from_output/1
    end
  end

  def logistic_derivative_from_output(y) do
    y * (1.0 - y)
  end

  def relu_derivative_from_output(y) when y <= 0 do
    0.0
  end

  def relu_derivative_from_output(_y) do
    1.0
  end

  def apply_activation_function_to_column(x, activation_function_id \\ :logistic)

  def apply_activation_function_to_column(x, activation_function_id)
      when activation_function_id == :softmax do
    [row_vector] = Matrix.transpose(x)
    exp_sum = Enum.reduce(row_vector, 0, fn element, acc -> acc + :math.exp(element) end)
    Enum.map(row_vector, fn element -> :math.exp(element) / exp_sum end) |> Matrix.column()
  end

  def apply_activation_function_to_column(x, activation_function_id) do
    Enum.map(x, fn [z] -> [Activation.function(activation_function_id).(z)] end)
  end
end

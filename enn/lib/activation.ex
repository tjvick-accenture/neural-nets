defmodule Activation do
  def function(a) do
    case a do
      :step -> &step_function/1
      :logistic -> &logistic_function/1
      :relu -> &relu_function/1
    end
  end

  def step_function(x) do
    if x >= 0, do: 1.0, else: 0.0
  end

  def logistic_function(x) do
    cond do
      x < -40 -> 0.0
      40 < x -> 1.0
      true -> 1 / (1 + :math.exp(-x))
    end
  end

  def relu_function(x) do
    if x <=0, do: 0.0, else: x
  end


  def apply_activation_function_to_column(x, :softmax) do
    [row_vector] = Matrix.transpose(x)
    exp_sum = Enum.reduce(row_vector, 0, fn element, acc -> acc + :math.exp(element) end)
    Enum.map(row_vector, fn element -> :math.exp(element) / exp_sum end) |> Matrix.column()
  end

  def apply_activation_function_to_column(x, activation_function_id) do
    apply_function_to_column(x, Activation.function(activation_function_id))
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

  def relu_derivative_from_output(y) do
    if y <= 0, do: 0.0, else: 1.0
  end


  def derivative_of_output_wrt_input(activation_output, :softmax) do
    a = hd(Matrix.transpose(activation_output))
    n = length(a)

    for ir <- 0..(n - 1) do
      for ic <- 0..(n - 1) do
        a_i = Enum.at(a, ir)
        a_j = Enum.at(a, ic)
        if ir == ic, do: a_i * (1 - a_i), else: -a_i * a_j
      end
    end
  end

  def derivative_of_output_wrt_input(activation_output, activation_id) do
    activation_output
    |> apply_function_to_column(Activation.derivative_from_output(activation_id))
    |> List.flatten()
    |> Matrix.diagonal()
  end


  defp apply_function_to_column(x, function) do
    Enum.map(x, fn [z] -> [function.(z)] end)
  end
end

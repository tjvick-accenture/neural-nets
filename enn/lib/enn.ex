defmodule ENN do
  require Logger

  def neuron_layer_output(input, weights, bias, activation_id \\ :logistic) do
    Matrix.multiply(weights, input)
    |> Matrix.add(bias)
    |> Activation.apply_activation_function_to_column(activation_id)
  end

  def d_activation_input_wrt_weights(input) do
    input
  end

  def d_loss_wrt_activation_output(activation_output, target_output, :absolute_error) do
    [y] = Matrix.transpose(activation_output)
    [t] = Matrix.transpose(target_output)

    [
      Enum.zip(y, t)
      |> Enum.map(fn {yi, ti} -> function_derivative_absolute_error(yi, ti) end)
    ]
  end

  def d_loss_wrt_activation_output(activation_output, target_output, :squared_error) do
    [y] = Matrix.transpose(activation_output)
    [t] = Matrix.transpose(target_output)

    [
      Enum.zip(y, t)
      |> Enum.map(fn {yi, ti} -> function_derivative_squared_error(yi, ti) end)
    ]
  end

  def d_loss_wrt_activation_output(activation_output, target_output, :categorical_crossentropy) do
    [y] = Matrix.transpose(activation_output)
    [t] = Matrix.transpose(target_output)

    [
      Enum.zip(y, t)
      |> Enum.map(fn {yi, ti} -> function_derivative_categorical_crossentropy(yi, ti) end)
    ]
  end

  def function_derivative_absolute_error(output, target) do
    error = output - target

    cond do
      error == 0 -> 0
      error > 0 -> 1
      error < 0 -> -1
    end
  end

  def function_derivative_squared_error(output, target) do
    output - target
  end

  def function_derivative_categorical_crossentropy(output, target) do
    cond do
      target == 0 -> 0
      output == 0 and target > 0 -> -1.0e100
      output == 0 and target < 0 -> 1.0e100
      true -> -target / output
    end
  end

  defp loss_gradient_chain_rule(
         loss_wrt_output,
         output_wrt_activation_input,
         activation_input_wrt_layer_input
       ) do
    Matrix.transpose(output_wrt_activation_input)
    |> Matrix.multiply(Matrix.transpose(loss_wrt_output))
    #    |> Matrix.multiply([[1]])  # truly a desired change in cost # Could put learning rate here instead
    |> Matrix.multiply(Matrix.transpose(activation_input_wrt_layer_input))
  end

  defp compute_weight_change(input, output, target, activation_id, loss_id) do
    # This could be rolled into the loss_gradient_chain_rule
    learning_rate = 0.2

    # Calculate gradient vector of loss wrt activation output
    l_a = d_loss_wrt_activation_output(output, target, loss_id)

    # Calculate gradient matrix of activation output wrt activation input
    a_z = Activation.derivative_of_output_wrt_input(output, activation_id)

    # Calculate gradient vector of activation input wrt weights (== input vector when fully connected)
    z_w = d_activation_input_wrt_weights(input)

    # Calculate gradient matrix of loss wrt weights by applying chain rule
    l_w = loss_gradient_chain_rule(l_a, a_z, z_w)

    # Calculate delta W using learning rate
    l_w |> Matrix.multiply_each(-learning_rate)
  end

  def backpropagate_once(input, weights, bias, target, activation_id, loss_id) do
    output = neuron_layer_output(input, weights, bias, activation_id)

    delta_w = compute_weight_change(input, output, target, activation_id, loss_id)

    Matrix.add(weights, delta_w)
  end

  def train(input_vectors, target_vectors, n_cycles, activation_id, loss_id) do
    m = length(hd(target_vectors))
    n = length(hd(input_vectors))
    weights = Matrix.random(m, n)
    bias = Matrix.zeros(m, 1)

    backpropagate_through_each_input_target_pair_n_times(
      input_vectors,
      target_vectors,
      weights,
      bias,
      n_cycles,
      activation_id,
      loss_id
    )
  end

  defp backpropagate_through_each_input_target_pair_n_times(
         _input_vectors,
         _target_vectors,
         weights,
         _bias,
         n_cycles,
         _activation_id,
         loss_id
       )
       when n_cycles == 0 do
    weights
  end

  defp backpropagate_through_each_input_target_pair_n_times(
         input_vectors,
         target_vectors,
         weights,
         bias,
         n_cycles,
         activation_id,
         loss_id
       )
       when n_cycles <= 1 do
    backpropagate_through_each_input_target_pair(
      input_vectors,
      target_vectors,
      weights,
      bias,
      activation_id,
      loss_id
    )
  end

  defp backpropagate_through_each_input_target_pair_n_times(
         input_vectors,
         target_vectors,
         weights,
         bias,
         n_cycles,
         activation_id,
         loss_id
       ) do
    new_weights =
      backpropagate_through_each_input_target_pair(
        input_vectors,
        target_vectors,
        weights,
        bias,
        activation_id,
        loss_id
      )

    backpropagate_through_each_input_target_pair_n_times(
      input_vectors,
      target_vectors,
      new_weights,
      bias,
      n_cycles - 1,
      activation_id,
      loss_id
    )
  end

  defp backpropagate_through_each_input_target_pair(
         [input_vector | input_vectors],
         [target_vector | _],
         weights,
         bias,
         activation_id,
         loss_id
       )
       when length(input_vectors) == 0 do
    backpropagate_once(input_vector, weights, bias, target_vector, activation_id, loss_id)
  end

  defp backpropagate_through_each_input_target_pair(
         [input_vector | input_vectors],
         [target_vector | target_vectors],
         weights,
         bias,
         activation_id,
         loss_id
       ) do
    new_weights =
      backpropagate_once(input_vector, weights, bias, target_vector, activation_id, loss_id)

    backpropagate_through_each_input_target_pair(
      input_vectors,
      target_vectors,
      new_weights,
      bias,
      activation_id,
      loss_id
    )
  end

  def test(input_vectors, target_vectors, weights, activation_id) do
    m = length(hd(target_vectors))
    bias = Matrix.zeros(m, 1)

    Enum.reduce(
      Enum.zip(input_vectors, target_vectors),
      {0, 0},
      fn {input_vector, target_vector}, {n_correct, n_incorrect} ->
        output_vector = neuron_layer_output(input_vector, weights, bias, activation_id)
        output_index = argmax(hd(Matrix.transpose(output_vector)))
        target_index = argmax(hd(Matrix.transpose(target_vector)))

        if output_index == target_index do
          {n_correct + 1, n_incorrect}
        else
          {n_correct, n_incorrect + 1}
        end
      end
    )
  end

  defp argmax(enum) do
    max_value = Enum.max(enum)
    Enum.find_index(enum, fn el -> el == max_value end)
  end
end

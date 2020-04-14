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

  def d_activation_output_wrt_activation_input(activation_output, :softmax) do
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

  def d_activation_output_wrt_activation_input(activation_output, activation_id) do
    activation_output
    |> Enum.map(fn [y] -> Activation.derivative_from_output(activation_id).(y) end)
    |> Matrix.diagonal()
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

  defp cost_gradient_function_L2(activation_derivative, input, output, target) do
    error = output - target
    activation_derivative * input * error
  end

  defp cost_gradient_function_L1(activation_derivative, input, output, target) do
    error = output - target
    slope = if error < 0, do: -1.0, else: 1.0
    activation_derivative * input * slope
  end

  defp cost_gradient_function_CECL(activation_derivative, input, output, target) do
    activation_derivative * input * target / output * -1.0
  end

  def backpropagate_once(input, weights, bias, target, activation_id, loss_id) do
    output = neuron_layer_output(input, weights, bias, activation_id)

    # 1. Calculate gradient vector of LOSS wrt activation output as function of TARGET vector and OUTPUT vector
    #    L_A = loss_wrt_activation_output(output, target)
    l_a = d_loss_wrt_activation_output(output, target, loss_id)

    # 2. Calculate gradient matrix of ACTIVATION output wrt activation input as function of OUTPUT vector
    #    A_Z = activation_output_wrt_activation_input(output)
    a_z = d_activation_output_wrt_activation_input(output, activation_id)

    # 3. Assume fully-connected weights matrix (:: gradient matrix of activation input wrt weights == INPUT vector)
    #    Z_W = activation_input_wrt_weights(input)
    z_w = d_activation_input_wrt_weights(input)

    # 4. Calculate gradient matrix of loss wrt weights from 1-3
    #    loss_wrt_weights = gradient_chain_rule(C_A, A_Z, Z_W)
    l_w =
      Matrix.transpose(a_z)
      |> Matrix.multiply(Matrix.transpose(l_a))
      |> Matrix.multiply([[-0.03]])
      |> Matrix.multiply(Matrix.transpose(z_w))

    # 5. Calculate delta W using LEARNING RATE
    #    delta_w = loss_wrt_weights |> Matrix.multiply_each(learning_Rate)
    delta_w = l_w

    # 6. Apply weight change to WEIGHTS matrix (Or accumulate weight change)
    Matrix.add(weights, delta_w)

    #    for {weights_row, [output_value], [target_value]} <- Enum.zip([weights, output, target]) do
    #      backpropagate_for_single_output_neuron(
    #        weights_row,
    #        output_value,
    #        target_value,
    #        input,
    #        activation_id
    #      )
    #    end
  end

  #  def backpropagate_for_single_output_neuron(
  #        weights_row,
  #        output_value,
  #        target_value,
  #        input,
  #        activation_id \\ :logistic
  #      ) do
  #    for {weight, [input_value]} <- Enum.zip(weights_row, input) do
  #      apply_weight_change(weight, input_value, output_value, target_value, activation_id)
  #    end
  #  end

  #  def apply_weight_change(weight, input, output, target, activation_id \\ :logistic) do
  #    weight + calculate_weight_change(input, output, target, activation_id)
  #  end

  #  def calculate_weight_change(input, output, target, activation_id \\ :logistic) do
  #    learning_rate = 0.1
  #    activation_derivative_function = Activation.derivative_from_output(activation_id)
  #    activation_derivative_value = activation_derivative_function.(output)
  #
  #    cost_derivative_wrt_weight =
  #      cost_gradient_function_CECL(activation_derivative_value, input, output, target)
  #
  #    -1.0 * learning_rate * cost_derivative_wrt_weight
  #  end

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

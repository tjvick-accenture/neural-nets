defmodule ENNTest do
  use ExUnit.Case
  require Logger
  doctest ENN

  describe "neuron_layer_output/3 as a perceptron as an AND gate" do
    test "returns 'true' for two 'true' inputs" do
      weights = [
        [2.0, 2.0]
      ]

      bias = [-3.0] |> Matrix.column()
      input = [1.0, 1.0] |> Matrix.column()

      assert ENN.neuron_layer_output(input, weights, bias, :step) === [[1.0]]
    end

    test "returns 'false' for two 'false' inputs" do
      weights = [
        [2.0, 2.0]
      ]

      bias = [-3.0] |> Matrix.column()
      input = [0.0, 0.0] |> Matrix.column()

      assert ENN.neuron_layer_output(input, weights, bias, :step) === [[0.0]]
    end

    test "returns 'false' for one 'true' and one 'false' input" do
      weights = [
        [2.0, 2.0]
      ]

      bias = [-3.0] |> Matrix.column()
      input = [0.0, 1.0] |> Matrix.column()

      assert ENN.neuron_layer_output(input, weights, bias, :step) === [[0.0]]
    end
  end

  describe "neuron_layer_output/3 as a perceptron as a 2x2 network" do
    test "produces 2 output values" do
      weights = [
        [1.0, 2.0],
        [-1.0, -2.0]
      ]

      bias = [5.0, 6.0] |> Matrix.column()
      input = [7.0, 8.0] |> Matrix.column()
      expected = [1.0, 0.0] |> Matrix.column()

      assert ENN.neuron_layer_output(input, weights, bias, :step) === expected
    end
  end

  describe "neuron_layer_output/3" do
    test "produces 0.5 for each output of a zero-weight, zero-bias 3x3 layer" do
      weights = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
      ]

      bias = [0, 0, 0] |> Matrix.column()
      input = [1, 2, 3] |> Matrix.column()
      expected = [0.5, 0.5, 0.5] |> Matrix.column()

      assert ENN.neuron_layer_output(input, weights, bias, :logistic) === expected
    end

    test "produces 1.0 for each output of a large-positive-weight, large-positive-bias layer" do
      weights = [
        [100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0],
        [0, 0, 0]
      ]

      bias = [0, 0, 100] |> Matrix.column()
      input = [1, 2, 3] |> Matrix.column()
      expected = [1.0, 1.0, 1.0] |> Matrix.column()

      assert ENN.neuron_layer_output(input, weights, bias, :logistic) === expected
    end
  end

  describe "backpropagate_once/6" do
    Logger.debug("*** BACKPROP ONCE ***")

    def compute_cost(output, target) do
      [[o1], [o2]] = output
      [[t1], [t2]] = target
      (o1 - t1) * (o1 - t1) + (o2 - t2) * (o2 - t2)
    end

    test "reduces network cost after 1 iteration" do
      weights = [
        [0.1, 0.2],
        [0.3, 0.4]
      ]

      bias = [0.0, 0.0] |> Matrix.column()
      input = [1.0, 2.0] |> Matrix.column()
      target = [1.0, 0.0] |> Matrix.column()

      output = ENN.neuron_layer_output(input, weights, bias, :relu)
      Logger.debug(inspect(output))
      cost = compute_cost(output, target)

      new_weights = ENN.backpropagate_once(input, weights, bias, target, :relu, :squared_error)

      Logger.debug(inspect(new_weights))
      Logger.debug(inspect(new_weights |> Matrix.subtract(weights)))

      new_output = ENN.neuron_layer_output(input, new_weights, bias, :relu)
      Logger.debug(inspect(new_output))

      new_cost = compute_cost(new_output, target)

      assert new_cost < cost
    end
  end

  describe "train/3" do
    test "returns a weights matrix of appropriate size" do
      input_vectors = [
        Matrix.column([1.0, 0.0, 0.0]),
        Matrix.column([0.0, 1.0, 0.0]),
        Matrix.column([0.0, 0.0, 1.0])
      ]

      target_vectors = [
        Matrix.column([1.0, 0.0, 0.0]),
        Matrix.column([0.0, 1.0, 0.0]),
        Matrix.column([0.0, 0.0, 1.0])
      ]

      weights = ENN.train(input_vectors, target_vectors, 1, :relu, :absolute_error)
      assert length(weights) == 3
      assert length(hd(weights)) == 3
    end
  end

  describe "d_activation_input_wrt_weights/1" do
    test "returns input vector as a column" do
      input = [-1, 0, 1, 2] |> Matrix.column()
      z_w = ENN.d_activation_input_wrt_weights(input)
      assert z_w == input
    end
  end

  describe "d_loss_wrt_activation_output/3" do
    test "returns loss derivative wrt to the activation output for an absolute error loss function" do
      activation_output = [1.0, 0.5, 0.0] |> Matrix.column()
      target_output = [1.0, 0.0, 0.5] |> Matrix.column()
      l_a = ENN.d_loss_wrt_activation_output(activation_output, target_output, :absolute_error)

      assert l_a == [[0, 1, -1]]
    end

    test "returns loss derivative wrt to the activation output for a squared error loss function" do
      activation_output = [1.0, 0.5, 0.0] |> Matrix.column()
      target_output = [1.0, 0.0, 0.5] |> Matrix.column()
      l_a = ENN.d_loss_wrt_activation_output(activation_output, target_output, :squared_error)

      assert l_a == [[0, 0.5, -0.5]]
    end

    test "returns loss derivative wrt to the activation output for a categorical crossentropy loss" do
      activation_output = [1.0, 0.5, 0.1, 0.0, 0.0] |> Matrix.column()
      target_output = [1.0, 0.0, 0.5, 0.5, 0.0] |> Matrix.column()

      l_a =
        ENN.d_loss_wrt_activation_output(
          activation_output,
          target_output,
          :categorical_crossentropy
        )

      assert l_a == [[-1.0, 0, -5.0, -1.0e100, 0.0]]
    end
  end
end

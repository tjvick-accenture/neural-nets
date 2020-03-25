defmodule ENNTest do
  use ExUnit.Case
  doctest ENN

  describe "perceptron_output/3 as an AND gate" do
    test "returns 'true' for two 'true' inputs" do
      weights = [
        [2.0, 2.0]
      ]

      bias = [[-3.0]]

      input = [
        [1.0],
        [1.0]
      ]

      assert ENN.perceptron_output(weights, bias, input) === [1.0]
    end

    test "returns 'false' for two 'false' inputs" do
      weights = [
        [2.0, 2.0]
      ]

      bias = [[-3.0]]

      input = [
        [0.0],
        [0.0]
      ]

      assert ENN.perceptron_output(weights, bias, input) === [0.0]
    end

    test "returns 'false' for one 'true' and one 'false' input" do
      weights = [
        [2.0, 2.0]
      ]

      bias = [[-3.0]]

      input = [
        [0.0],
        [1.0]
      ]

      assert ENN.perceptron_output(weights, bias, input) === [0.0]
    end
  end

  describe "perceptron_output/3 as a 2x2 network" do
    test "produces 2 output values" do
      weights = [
        [1.0, 2.0],
        [-1.0, -2.0]
      ]

      bias = [
        [5.0],
        [6.0]
      ]

      input = [
        [7.0],
        [8.0]
      ]

      assert ENN.perceptron_output(weights, bias, input) === [1.0, 0.0]
    end
  end

  describe "neuron_layer_output/3" do
    test "produces 0.5 for each output of a zero-weight, zero-bias 3x3 layer" do
      weights = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
      ]

      bias = Matrix.matrix([0, 0, 0]) |> Matrix.transpose

      input = Matrix.matrix([1, 2, 3]) |> Matrix.transpose

      assert ENN.neuron_layer_output(input, weights, bias) === [0.5, 0.5, 0.5]
    end

    test "produces 1.0 for each output of a large-positive-weight, large-positive-bias layer" do
      weights = [
        [100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0],
        [0, 0, 0]
      ]

      bias = Matrix.matrix([0, 0, 100]) |> Matrix.transpose

      input = Matrix.matrix([1, 2, 3]) |> Matrix.transpose

      assert ENN.neuron_layer_output(input, weights, bias) === [1.0, 1.0, 1.0]
    end
  end
end

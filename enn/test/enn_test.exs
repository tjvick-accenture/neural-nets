defmodule ENNTest do
  use ExUnit.Case
  doctest ENN

  describe("perceptron_output/3 as an AND gate") do
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

  describe("step_function/1") do
    test "returns 1 when input is positive" do
      assert ENN.step_function(0.5) === 1.0
    end

    test "returns 0 when input is negative" do
      assert ENN.step_function(-0.5) === 0.0
    end

    test "returns 1 when input is zero" do
      assert ENN.step_function(0.0) === 1.0
    end
  end
end

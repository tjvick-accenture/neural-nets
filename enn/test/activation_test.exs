defmodule ActivationTest do
  use ExUnit.Case
  doctest Activation

  describe "logistic_function/1" do
    test "returns 0.5 when input is 0" do
      assert Activation.logistic_function(0) === 0.5
    end

    test "approaches 1 when input is greater than zero" do
      assert Activation.logistic_function(10) > 0.5
      assert Activation.logistic_function(10) < 1.0
    end

    test "is practically 1 when input is large" do
      assert Activation.logistic_function(100) > 1.0 - 1.0e-16
      assert Activation.logistic_function(100) <= 1.0
    end

    test "returns 1 for very large positive values" do
      assert Activation.logistic_function(1.0e16) === 1.0
    end

    test "approaches 0 when input is less than zero" do
      assert Activation.logistic_function(-10) < 0.5
      assert Activation.logistic_function(-10) > 0.0
    end

    test "is practically 1 when input is negative and large" do
      assert Activation.logistic_function(-100) < 1.0e-16
      assert Activation.logistic_function(-100) >= 0.0
    end

    test "returns 0 for very large negative numbers" do
      assert Activation.logistic_function(-1.0e16) === 0.0
    end
  end

  describe "step_function/1" do
    test "returns 1 when input is positive" do
      assert Activation.step_function(0.5) === 1.0
    end

    test "returns 0 when input is negative" do
      assert Activation.step_function(-0.5) === 0.0
    end

    test "returns 1 when input is zero" do
      assert Activation.step_function(0.0) === 1.0
    end
  end
end

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

  describe "relu_function/1" do
    test "returns input when input is positive" do
      assert Activation.relu_function(0.7) == 0.7
      assert Activation.relu_function(7.0) == 7.0
      assert Activation.relu_function(70.0) == 70.0
    end

    test "returns zero when input is negative" do
      assert Activation.relu_function(-0.7) == 0.0
      assert Activation.relu_function(-7.0) == 0.0
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

  describe "function/1" do
    test "returns logistic function when prompted" do
      assert Activation.function(:logistic).(0.0) === Activation.logistic_function(0.0)
      assert Activation.function(:logistic).(1.0) === Activation.logistic_function(1.0)
      assert Activation.function(:logistic).(5.0) === Activation.logistic_function(5.0)
      assert Activation.function(:logistic).(-2.0) === Activation.logistic_function(-2.0)
    end

    test "returns step function when prompted" do
      assert Activation.function(:step).(0.0) === Activation.step_function(0.0)
      assert Activation.function(:step).(1.0) === Activation.step_function(1.0)
      assert Activation.function(:step).(5.0) === Activation.step_function(5.0)
      assert Activation.function(:step).(-2.0) === Activation.step_function(-2.0)
    end

    test "returns relu function when prompted" do
      assert Activation.function(:relu).(0.0) === Activation.relu_function(0.0)
      assert Activation.function(:relu).(5.0) === Activation.relu_function(5.0)
      assert Activation.function(:relu).(-2.0) === Activation.relu_function(-2.0)
    end
  end

  describe "logistic_derivative_from_output/1" do
    test "computes the result as x(1-x)" do
      assert Activation.logistic_derivative_from_output(0.0) === 0.0
      assert Activation.logistic_derivative_from_output(1.0) === 0.0
      assert Activation.logistic_derivative_from_output(2.0) === -2.0
      assert Activation.logistic_derivative_from_output(-1.0) === -2.0
      assert Activation.logistic_derivative_from_output(-2.0) === -6.0
      assert Activation.logistic_derivative_from_output(3.0) === -6.0
    end
  end

  describe "relu_derivative_from_output/1" do
    test "returns 1 when input is positive" do
      assert Activation.relu_derivative_from_output(1.0) === 1.0
      assert Activation.relu_derivative_from_output(2.0) === 1.0
      assert Activation.relu_derivative_from_output(3.0) === 1.0
    end

    test "returns zero when input is less than or equal to zero" do
      assert Activation.relu_derivative_from_output(0.0) === 0.0
      assert Activation.relu_derivative_from_output(-1.0) === 0.0
      assert Activation.relu_derivative_from_output(-2.0) === 0.0
    end
  end

  describe "derivative_from_output/1" do
    test "returns derivative of logistic function" do
      assert Activation.derivative_from_output(:logistic).(0.0) ===
               Activation.logistic_derivative_from_output(0.0)

      assert Activation.derivative_from_output(:logistic).(1.0) ===
               Activation.logistic_derivative_from_output(1.0)

      assert Activation.derivative_from_output(:logistic).(2.0) ===
               Activation.logistic_derivative_from_output(2.0)

      assert Activation.derivative_from_output(:logistic).(3.0) ===
               Activation.logistic_derivative_from_output(3.0)

      assert Activation.derivative_from_output(:logistic).(-1.0) ===
               Activation.logistic_derivative_from_output(-1.0)

      assert Activation.derivative_from_output(:logistic).(-2.0) ===
               Activation.logistic_derivative_from_output(-2.0)

      assert Activation.derivative_from_output(:logistic).(-3.0) ===
               Activation.logistic_derivative_from_output(-3.0)
    end

    test "returns derivative of relu function" do
      assert Activation.derivative_from_output(:relu).(0.0) ===
               Activation.relu_derivative_from_output(0.0)

      assert Activation.derivative_from_output(:relu).(2.0) ===
               Activation.relu_derivative_from_output(2.0)

      assert Activation.derivative_from_output(:relu).(-3.0) ===
               Activation.relu_derivative_from_output(-3.0)
    end
  end

  describe "apply_activation_function_to_column/2" do
    test "applies relu to whole column when relu is specified" do
      x = [-2, -1, 0, 1, 2, 3] |> Matrix.column()
      output = Activation.apply_activation_function_to_column(x, :relu)

      expected_output = [0, 0, 0, 1, 2, 3] |> Matrix.column()
      assert output == expected_output
    end

    test "applies softmax to whole column when softmax is specified" do
      e = 2.71828182846
      x = [0, 1, 1] |> Matrix.column()
      output = Activation.apply_activation_function_to_column(x, :softmax)

      expected_output =
        [0.15536240349696362, 0.4223187982515182, 0.4223187982515182]
        |> Matrix.column()

      assert output == expected_output
    end
  end
end

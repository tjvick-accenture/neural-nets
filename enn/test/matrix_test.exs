defmodule MatrixTest do
  use ExUnit.Case
  doctest Matrix

  test "multiply performs multiplication of [1x1] with [1x1]" do
    assert Matrix.multiply([[1]], [[1]]) == [[1]]
  end

  test "multiply performs multiplication of [2x2] with [2x1]" do
    assert Matrix.multiply([[1, 2], [3, 4]], [[1], [1]]) == [[3], [7]]
  end

  test "multiply performs multiplication of [2x3] with [3x1]" do
    assert Matrix.multiply([[1, 2, 3], [4, 5, 6]], [[1], [1], [1]]) == [[6], [15]]
  end

  test "multiply performs multiplication of [2x2] with [2x2]" do
    m = [[1, 2], [3, 4]]
    assert Matrix.multiply(m, m) == [[7, 10], [15, 22]]
  end

  test "transpose does nothing to 1x1" do
    assert Matrix.transpose([[1]]) == [[1]]
  end

  test "transpose transposes 1x2 into 2x1" do
    assert Matrix.transpose([[1], [1]]) == [[1, 1]]
  end

  test "transpose transposes 2x1 into 1x2" do
    assert Matrix.transpose([[1, 1]]) == [[1], [1]]
  end

  test "transpose transposes 2x2" do
    assert Matrix.transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
  end

  test "transpose transposes 2x3 into 3x2" do
    assert Matrix.transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
  end
end

defmodule MatrixTest do
  use ExUnit.Case
  doctest Matrix

  describe "Matrix.add/2" do
    test "adds [1x1] to [1x1]" do
      assert Matrix.add([[1]], [[2]]) == [[3]]
    end

    test "adds [1x2] to [1x2]" do
      assert Matrix.add([[1, 2]], [[3, 4]]) == [[4, 6]]
    end

    test "adds [2x2] to [2x2]" do
      a = [
        [1, 2],
        [3, 4]
      ]

      b = a

      y = [
        [2, 4],
        [6, 8]
      ]

      assert Matrix.add(a, b) == y
    end
  end

  describe "Matrix.subtract/2" do
    test "subtracts [1x1] and [1x1]" do
      assert Matrix.subtract([[1]], [[0.5]]) === [[0.5]]
    end
  end

  describe "Matrix.multiply/2" do
    test "performs multiplication of [1x1] with [1x1]" do
      assert Matrix.multiply([[1]], [[1]]) == [[1]]
    end

    test "performs multiplication of [1x2] with [2x1]" do
      a = [
        [1, 2]
      ]

      b = [
        [1],
        [1]
      ]

      y = [
        [3]
      ]

      assert Matrix.multiply(a, b) == y
    end

    test "performs multiplication of [2x2] with [2x1]" do
      a = [
        [1, 2],
        [3, 4]
      ]

      b = [
        [1],
        [1]
      ]

      y = [
        [3],
        [7]
      ]

      assert Matrix.multiply(a, b) == y
    end

    test "performs multiplication of [2x3] with [3x1]" do
      a = [
        [1, 2, 3],
        [4, 5, 6]
      ]

      b = [
        [1],
        [1],
        [1]
      ]

      y = [
        [6],
        [15]
      ]

      assert Matrix.multiply(a, b) == y
    end

    test "performs multiplication of [2x2] with [2x2]" do
      a = [
        [1, 2],
        [3, 4]
      ]

      b = a

      y = [
        [7, 10],
        [15, 22]
      ]

      assert Matrix.multiply(a, b) == y
    end

    test "performs multiplication of [2x1] with [1x2]" do
      a = [
        [1],
        [2]
      ]

      b = [
        [1, 2]
      ]

      y = [
        [1, 2],
        [2, 4]
      ]

      assert Matrix.multiply(a, b) == y
    end
  end

  describe "Matrix.transpose/1" do
    test "does nothing to 1x1" do
      assert Matrix.transpose([
               [1]
             ]) == [
               [1]
             ]
    end

    test "transposes 1x2 into 2x1" do
      assert Matrix.transpose([
               [1],
               [1]
             ]) == [
               [1, 1]
             ]
    end

    test "transposes 2x1 into 1x2" do
      assert Matrix.transpose([
               [1, 1]
             ]) == [
               [1],
               [1]
             ]
    end

    test "transposes 2x2" do
      assert Matrix.transpose([
               [1, 2],
               [3, 4]
             ]) == [
               [1, 3],
               [2, 4]
             ]
    end

    test "transposes 2x3 into 3x2" do
      assert Matrix.transpose([
               [1, 2, 3],
               [4, 5, 6]
             ]) == [
               [1, 4],
               [2, 5],
               [3, 6]
             ]
    end
  end

  describe "Matrix.multiply_each/2" do
    test "Multiplies each element of a 1x1 by scalar" do
      assert Matrix.multiply_each([[3.0]], 2.0) == [[6.0]]
    end

    test "Multiplies each element of a 1x2 by scalar" do
      assert Matrix.multiply_each([[3.0, 3.5]], 2.0) == [[6.0, 7.0]]
    end

    test "Multiplies each element of a 2x2 by scalar" do
      assert Matrix.multiply_each([[2.0, 3.0], [4.0, 5.0]], 2.0) == [[4.0, 6.0], [8.0, 10.0]]
    end
  end

  describe "Matrix.divide_each/2" do
    test "Divides each element of a 1x1 by divisor" do
      assert Matrix.divide_each([[6.0]], 2.0) == [[3.0]]
    end

    test "Divides each element of a 1x2 by divisor" do
      assert Matrix.divide_each([[6.0, 7.0]], 2.0) == [[3.0, 3.5]]
    end

    test "Divides each element of a 2x2 by divisor" do
      assert Matrix.divide_each([[4.0, 5.0], [6.0, 7.0]], 2.0) == [[2.0, 2.5], [3.0, 3.5]]
    end
  end

  describe "Matrix.matrix/1" do
    test "Converts a single integer to a matrix" do
      assert Matrix.matrix(1) == [[1]]
    end

    test "Converts a single float to a matrix" do
      assert Matrix.matrix(1.0) === [[1.0]]
    end

    test "Converts a list to a matrix" do
      assert Matrix.matrix([1, 2, 3]) === [[1, 2, 3]]
    end

    test "Does not modify a list of lists" do
      a = [[1, 2, 3], [4, 5, 6]]
      assert Matrix.matrix(a) === a
    end
  end

  describe "Matrix.negate/1" do
    test "makes single positive value negative" do
      assert Matrix.negate([[1.0]]) === [[-1.0]]
    end

    test "makes single negative value positive" do
      assert Matrix.negate([[-1.0]]) === [[1.0]]
    end

    test "makes multiple positive values negative" do
      assert Matrix.negate([[1.0, 2.0], [3.0, 4.0]]) === [[-1.0, -2.0], [-3.0, -4.0]]
    end

    test "makes multiple negative values positive" do
      m = [[1.0, 2.0], [3.0, 4.0]]
      assert Matrix.negate(Matrix.negate(m)) === m
    end
  end

  describe "Matrix.column/1" do
    test "converts list to column matrix" do
      assert Matrix.column([0.0, 1.1, 2.2]) === [
               [0.0],
               [1.1],
               [2.2]
             ]
    end
  end

  describe "Matrix.diagonal/1" do
    test "converts list to matrix with values on the diagonal" do
      assert Matrix.diagonal([0.1, 2.3, 4.5]) == [
               [0.1, 0.0, 0.0],
               [0.0, 2.3, 0.0],
               [0.0, 0.0, 4.5]
             ]
    end
  end

  describe "Matrix.random/2" do
    test "generates a 1x1 matrix of a float between 0.0 and 1.0" do
      [[a]] = Matrix.random(1, 1)
      assert a <= 1.0 and a >= 0.0
    end

    test "generates a 1x2 matrix using random values" do
      [[a, b]] = Matrix.random(1, 2)
      assert a <= 1.0 and a >= 0.0
      assert b <= 1.0 and b >= 0.0
      assert a != b
    end

    test "generates a 3x5 matrix" do
      a = Matrix.random(3, 5)
      assert length(a) == 3
      assert length(hd(a)) == 5
    end
  end

  describe "Matrix.zeros/2" do
    test "generates a 1x1 matrix of 0.0s" do
      [[a]] = Matrix.zeros(1, 1)
      assert a == 0.0
    end

    test "generates a 1x2 matrix of 0.0s" do
      [[a, b]] = Matrix.zeros(1, 2)
      assert a == 0.0
      assert b == 0.0
    end

    test "generates a 5x3 matrix of 0.0s" do
      x = Matrix.zeros(5, 3)
      assert length(x) == 5
      assert length(hd(x)) == 3
    end
  end
end

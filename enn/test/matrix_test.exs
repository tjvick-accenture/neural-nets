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
end

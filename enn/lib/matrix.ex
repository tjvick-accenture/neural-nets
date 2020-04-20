defmodule Matrix do
  def add(a, b) do
    Enum.zip(a, b)
    |> Enum.map(&sum_tuple_of_vectors/1)
  end

  def subtract(a, b) do
    add(a, Matrix.negate(b))
  end

  defp sum_tuple_of_vectors({a, b}) do
    Enum.zip(a, b)
    |> Enum.map(&sum_tuple/1)
  end

  defp sum_tuple({a, b}) do
    a + b
  end

  def multiply(a, b) do
    transpose(b)
    |> Enum.map(&multiply_matrix_by_vector(a, &1))
    |> transpose
  end

  defp multiply_matrix_by_vector(m, v) do
    Enum.map(m, &vector_dot(&1, v))
  end

  defp vector_dot(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {ai, bi} -> ai * bi end)
    |> Enum.sum()
  end

  def negate(x) do
    for row <- x do
      for element <- row do
        -element
      end
    end
  end

  def multiply_each(m, s) do
    for row <- m do
      for element <- row do
        element * s
      end
    end
  end

  def divide_each(m, d) do
    for row <- m do
      for element <- row do
        element / d
      end
    end
  end

  def transpose(x) do
    List.zip(x)
    |> Enum.map(&Tuple.to_list(&1))
  end

  def matrix(x) when is_integer(x) or is_float(x) do
    [[x]]
  end

  def matrix(x) when is_list(x) do
    if List.flatten(x) == x do
      [x]
    else
      x
    end
  end

  def column(x) when is_list(x) do
    Matrix.transpose([x])
  end

  def diagonal(x) when is_list(x) do
    m = length(x)

    for ir <- 0..(m - 1) do
      for ic <- 0..(m - 1) do
        if ic == ir, do: Enum.at(x, ic), else: 0.0
      end
    end
  end

  def random(m, n) do
    _ = :rand.seed(:exs1024, {1, 123_534, 345_345})

    for _m <- 1..m do
      for _n <- 1..n do
        :rand.uniform()
      end
    end
  end

  def zeros(m, n) do
    for _m <- 1..m do
      for _n <- 1..n do
        0
      end
    end
  end
end

defmodule Matrix do
  def add(a, b) do
    Enum.zip(a, b)
    |> Enum.map(&sum_tuple_of_vectors/1)
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
end

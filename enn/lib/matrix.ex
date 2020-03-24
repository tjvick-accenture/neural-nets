defmodule Matrix do
  def multiply(w, x) do
    xt = transpose(x)

    yT = Enum.map(xt, fn(xi) -> multiply_matrix_by_vector(w, xi) end)

    transpose(yT)
  end

  defp multiply_matrix_by_vector(w, v) do
    Enum.map(w, fn(wi) -> vector_dot(wi, v) end)
  end

  defp vector_dot(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {ai, bi} -> ai * bi end)
    |> Enum.reduce(0, fn x, acc -> x + acc end)
  end

  def transpose(x) do
    Enum.map(List.zip(x), &Tuple.to_list(&1))
  end
end

defmodule Activation do
  def step_function(x) when x >= 0 do
    1.0
  end

  def step_function(_x) do
    0.0
  end

  def logistic_function(x) when 40 < x do
    1.0
  end

  def logistic_function(x) when -40 > x do
    0.0
  end

  def logistic_function(x) do
    1 / (1 + :math.exp(-x))
  end
end

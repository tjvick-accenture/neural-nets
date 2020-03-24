defmodule ENNTest do
  use ExUnit.Case
  doctest ENN

  test "greets the world" do
    assert ENN.hello() == :world
  end
end

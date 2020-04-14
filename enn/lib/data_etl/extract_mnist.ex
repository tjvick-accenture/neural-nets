defmodule ExtractMNIST do
  require File
  require Path
  require Logger
  require Matrix

  def extract_input_vectors_from_file(filename) do
    {:ok, contents} = File.read(filename)

    <<
      _magic_number::32,
      _n_images::32,
      n_rows::32,
      n_columns::32,
      image_data::binary
    >> = contents

    image_size = n_rows * n_columns

    Enum.chunk_every(:binary.bin_to_list(image_data), image_size)
    |> Enum.map(&Matrix.column/1)
    |> Enum.map(&Matrix.divide_each(&1, 255.0))
  end

  def extract_input_vectors_from_file(filename, n_samples) do
    {:ok, contents} = File.read(filename)

    <<
      _magic_number::32,
      _n_images::32,
      n_rows::32,
      n_columns::32,
      image_data::binary
    >> = contents

    image_size = n_rows * n_columns

    image_binary_size = image_size * n_samples

    <<
      data_of_interest::binary-size(image_binary_size),
      _remainder::binary
    >> = image_data

    Enum.chunk_every(:binary.bin_to_list(data_of_interest), image_size)
    |> Enum.map(&Matrix.column/1)
    |> Enum.map(&Matrix.divide_each(&1, 255.0))
  end

  def extract_target_vectors_from_file(filename) do
    {:ok, contents} = File.read(filename)

    <<
      _magic_number::32,
      _n_labels::32,
      label_data::binary
    >> = contents

    :binary.bin_to_list(label_data)
    |> Enum.map(&one_hot_encode(&1, 10))
    |> Enum.map(&Matrix.column(&1))
  end

  def extract_target_vectors_from_file(filename, n_samples) do
    {:ok, contents} = File.read(filename)

    <<
      _magic_number::32,
      _n_labels::32,
      label_data_of_interest::binary-size(n_samples),
      _remainder::binary
    >> = contents

    :binary.bin_to_list(label_data_of_interest)
    |> Enum.map(&one_hot_encode(&1, 10))
    |> Enum.map(&Matrix.column(&1))
  end

  defp one_hot_encode(value, len) do
    List.duplicate(0.0, len) |> List.replace_at(value, 1.0)
  end
end

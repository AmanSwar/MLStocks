from data_prep.data_load import prepare_data , dataframe_to_tensor

import pandas as pd
import torch

test_file = "data/NIFTY_5_years.csv"


def test_prepare_data():

  df = prepare_data(test_file)
  
  assert type(df) == pd.DataFrame


def test_convert_to_tensor():

  df = prepare_data(test_file)

  tensorA = dataframe_to_tensor(df)

  assert type(tensorA) == torch.Tensor




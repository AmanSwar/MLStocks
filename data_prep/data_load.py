import pandas as pd

from data_prep.basic_feat import add_daily_return , indexify_date
from data_prep.features import add_all_indicators

def prepare_data(path : str) -> pd.DataFrame:
  df = pd.read_csv(path)

  df = indexify_date(df)
  df = add_daily_return(df)

  return add_all_indicators(df)



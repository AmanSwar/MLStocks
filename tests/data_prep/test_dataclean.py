from data_prep.data_clean import clean_indicator
from data_prep.data_load import prepare_data

import pandas as pd
import pytest

test_dir = "/home/aman/code/ml_fr/ml_stocks/data/NIFTY_5_years.csv"

test_df = prepare_data(test_dir)


@pytest.mark.filterwarnings("FutureWarning")
def test_data_clean(df : pd.DataFrame):

  clean_df = clean_indicator(df)

  resulting_colmns = [
    "Adj Close",
    "Close",
    "Dividends",
    "High",
    "Low",
    "Open",
    "Stock Splits",
    "Volume",
    "Daily_Return",
    "Log_Return",
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "SMA_100",
    "SMA_200",
    "EMA_5",
    "EMA_10",
    "EMA_12",
    "EMA_20",
    "EMA_26",
    "EMA_50",
    "EMA_100",
    "EMA_200",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "RSI_14",
    "Sto_%K",
    "Sto_%D",
    "Williams_%R_14",
    "CCI_20",
    "ROC_12",
    "Momentum_12",
    "CMO_14",
    "Ultimate_Osc",
    "OBV",
    "CMF_20",
    "ADL",
    "VPT",
    "VO_short_10",
    "VO_long_20",
    "Volume_Osc",
    "MFI_14",
    "ForceIndex_1",
    "ForceIndex_EMA_13",
    "TR",
    "ATR_14",
    "BB_MID_20",
    "BB_STD_20",
    "BB_UPPER_20",
    "BB_LOWER_20",
    "BB_Width",
    "Donchian_High_20",
    "Donchian_Low_20",
    "Donchian_Mid_20",
    "VWAP_cum",
    "HA_Open",
    "HA_Close",
    "HA_High",
    "HA_Low",
    "+DI_14",
    "-DI_14",
    "ADX_14",
    "Aroon_Up_25",
    "Aroon_Down_25",
    "Vortex_Pos_14",
    "Vortex_Neg_14",
]

  assert type(clean_df) == pd.DataFrame
  bool_arr = clean_df.columns == resulting_colmns
  assert bool_arr.all() == True

test_data_clean(test_df)

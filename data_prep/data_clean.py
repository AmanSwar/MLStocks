import pandas as pd
import numpy as np

from typing import List

def clean_indicator(
    df : pd.DataFrame,
    feature_cols : List[str] | None = None,
    drop_col_frac_threshold : float = 0.2
) -> pd.DataFrame:
  
  """
  function to clean dataframe (remove all NaN values without lookahead)
  args:
    df : pandas dataframe
    feature_cols : list of colms to treat as feats. Default -> None => use all colmns
    drop_col_frac_threshold : drop the colmns with total NaN values greater than this
  """

  df = df.copy()

  if feature_cols is None:

    exclude = {
            "Adj Close",
            "Dividends",
            "Stock Splits",
        }  # keep Close/Open/High/Low/Volume
    
    feature_cols = [c for c in df.columns if c not in exclude]

  #compute first valid and last valid positions
  n = len(df)
  first_positions = {}
  last_positions = {}

  for c in feature_cols:
    fv = df[c].first_valid_index()
    lv = df[c].last_valid_index()
    first_positions[c] = df.index.get_loc(fv) if fv is not None else n
    last_positions[c] = df.index.get_loc(lv) if lv is not None else -1

  #triming window logic -> remove head warmi
  #find features where all have values
  start_pos = max(
        first_positions.values()
    )
  #find last position where all features have values
  end_pos = min(last_positions.values())  
  
  if start_pos >= end_pos:
    # not enough overlap: as fallback, choose start = median of first positions, end = max of last positions
    start_pos = int(np.median(list(first_positions.values())))
    end_pos = int(np.median([pos for pos in last_positions.values() if pos >= 0]))

  df_trim = df.iloc[start_pos : (end_pos + 1)].copy()

  frac_nans = df_trim.isna().mean()
  drop_cols = frac_nans[frac_nans > drop_col_frac_threshold].index.tolist()
  # don't drop imp price columns
  essential = {"Open", "High", "Low", "Close", "Volume"}
  
  drop_cols = [c for c in drop_cols if c not in essential]
  
  df_trim = df_trim.drop(columns=drop_cols)

  df_imputed = df_trim.fillna(method="ffill") # type: ignore
  
  medians = df_imputed.median()
  df_imputed = df_imputed.fillna(medians)


  return df_imputed
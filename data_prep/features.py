import pandas as pd
import numpy as np

def _check_price_cols(df):
  required = ["Open", "High", "Low", "Close", "Volume"]
  missing = [c for c in required if c not in df.columns]
  if missing:
      raise ValueError(f"Missing required columns: {missing}")


def _rma(series: pd.Series, n: int):
  """Wilder's moving average (RMA)."""

  series = series.copy().astype(float)
  
  out = pd.Series(np.nan, index=series.index)
  
  if len(series) < n:
      return out
  
  out.iloc[n - 1] = series.iloc[:n].mean()
  
  alpha = 1.0 / n
  
  for i in range(n, len(series)):
      out.iat[i] = out.iat[i - 1] * (1 - alpha) + series.iat[i] * alpha
  
  return out


def _true_range(df):
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
  """
  Add Daily_Return and Log_Return 
  """
  df = df.copy()

  _check_price_cols(df)
  
  cols = ["Open", "High", "Low", "Close", "Volume"]
  df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

  df["Daily_Return"] = df["Close"].pct_change()
  
  df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
  
  return df

def add_trend_indicators(
    df: pd.DataFrame,
    sma_periods=(5, 10, 20, 50, 100, 200),
    ema_periods=(5, 10, 12, 20, 26, 50, 100, 200),
) -> pd.DataFrame:
  """
  Add SMA, EMA, MACD (12/26) and MACD signal/hist
  """
  df = df.copy()
  _check_price_cols(df)
  
  for p in sma_periods:
      df[f"SMA_{p}"] = df["Close"].rolling(window=p, min_periods=1).mean()
  
  for p in ema_periods:
      df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
  
  ema12 = df["Close"].ewm(span=12, adjust=False).mean()
  ema26 = df["Close"].ewm(span=26, adjust=False).mean()
  
  
  df["MACD"] = ema12 - ema26
  df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
  df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
  
  return df

def add_momentum_indicators(
    df: pd.DataFrame, rsi_n=14, sto_k=14, sto_d=3, cci_n=20, roc_n=12
) -> pd.DataFrame:
    """
    add common momentum indicators: 
    -RSI 
    -Stochastic %K/%D
    -Williams %R
    -CCI
    -ROC
    -Momentum
    -CMO
    -Ultimate Oscillator.
    """
    df = df.copy()
    _check_price_cols(df)

    close = df["Close"]

    # RSI (Wilder smoothing)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    df[f"RSI_{rsi_n}"] = 100 - (100 / (1 + (_rma(up, rsi_n) / _rma(down, rsi_n))))

    # Stochastic %K and %D
    low_min = df["Low"].rolling(window=sto_k).min()
    high_max = df["High"].rolling(window=sto_k).max()
    df["Sto_%K"] = 100 * (close - low_min) / (high_max - low_min)
    df["Sto_%D"] = df["Sto_%K"].rolling(window=sto_d).mean()

    # Williams %R
    df[f"Williams_%R_{sto_k}"] = -100 * (high_max - close) / (high_max - low_min)

    # CCI
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma_tp = tp.rolling(cci_n).mean()
    mad = tp.rolling(cci_n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df[f"CCI_{cci_n}"] = (tp - sma_tp) / (0.015 * mad)

    # ROC and Momentum
    df[f"ROC_{roc_n}"] = close.pct_change(periods=roc_n)
    df[f"Momentum_{roc_n}"] = close - close.shift(roc_n)

    # CMO (Chande Momentum Oscillator)
    up_sum = delta.clip(lower=0).rolling(rsi_n).sum()
    down_sum = -delta.clip(upper=0).rolling(rsi_n).sum()
    df[f"CMO_{rsi_n}"] = 100 * (up_sum - down_sum) / (up_sum + down_sum)

    # Ultimate Oscillator (7,14,28 default)
    def _ultimate_osc(df_, s1=7, s2=14, s3=28):
        bp = df_["Close"] - df_[["Low", "Close"]].shift(1).min(axis=1)
        tr = _true_range(df_)
        avg1 = bp.rolling(s1).sum() / tr.rolling(s1).sum()
        avg2 = bp.rolling(s2).sum() / tr.rolling(s2).sum()
        avg3 = bp.rolling(s3).sum() / tr.rolling(s3).sum()
        return 100 * ((4 * avg1) + (2 * avg2) + (1 * avg3)) / (4 + 2 + 1)

    df["Ultimate_Osc"] = _ultimate_osc(df)
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based indicators: 
    = OBV
    - CMF
    - ADL
    - VPT
    - Volume Oscillator
    - MFI
    - Force Index.
    """
    df = df.copy()
    _check_price_cols(df)

    # OBV
    df["OBV"] = ((np.sign(df["Close"].diff()) * df["Volume"]).fillna(0)).cumsum()
    # Money Flow Multiplier and Chaikin Money Flow
    mf_mult = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (
        df["High"] - df["Low"]
    )

    mf_mult = mf_mult.replace([np.inf, -np.inf], 0).fillna(0)
    mf_volume = mf_mult * df["Volume"]

    df["CMF_20"] = mf_volume.rolling(20).sum() / df["Volume"].rolling(20).sum()

    # ADL
    df["ADL"] = mf_volume.cumsum()

    # VPT
    df["VPT"] = (df["Volume"] * df["Close"].pct_change()).fillna(0).cumsum()

    # Volume Oscillator (short=10, long=20)
    df["VO_short_10"] = df["Volume"].rolling(10).mean()

    df["VO_long_20"] = df["Volume"].rolling(20).mean()

    df["Volume_Osc"] = (df["VO_short_10"] - df["VO_long_20"]) / df["VO_long_20"]

    # MFI (Money Flow Index)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    mf = tp * df["Volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0)
    neg_mf = mf.where(tp < tp.shift(1), 0)
    df["MFI_14"] = 100 - (
        100 / (1 + (pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum()))
    )

    # Force Index
    df["ForceIndex_1"] = df["Close"].diff(1) * df["Volume"]
    df["ForceIndex_EMA_13"] = df["ForceIndex_1"].ewm(span=13, adjust=False).mean()
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 
    - ATR
    - Bollinger Bands 
    - Donchian channels
    """
    df = df.copy()
    _check_price_cols(df)

    tr = _true_range(df)

    df["TR"] = tr
    df["ATR_14"] = tr.rolling(14).mean()

    # Bollinger Bands (20,2)
    n = 20
    df["BB_MID_20"] = df["Close"].rolling(n).mean()
    df["BB_STD_20"] = df["Close"].rolling(n).std()
    df["BB_UPPER_20"] = df["BB_MID_20"] + 2 * df["BB_STD_20"]
    df["BB_LOWER_20"] = df["BB_MID_20"] - 2 * df["BB_STD_20"]
    df["BB_Width"] = (df["BB_UPPER_20"] - df["BB_LOWER_20"]) / df["BB_MID_20"]

    # Donchian (20)
    dc = 20
    df["Donchian_High_20"] = df["High"].rolling(dc).max()
    df["Donchian_Low_20"] = df["Low"].rolling(dc).min()
    df["Donchian_Mid_20"] = (df["Donchian_High_20"] + df["Donchian_Low_20"]) / 2.0
    return df


# made by chatgpt because I have no idea how formula for these even works
def add_hybrid_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VWAP (cumulative), Heikin-Ashi candles, ADX, Aroon, Vortex.
    Each is implemented in a focused manner.
    """
    df = df.copy()
    _check_price_cols(df)
    # VWAP cumulative
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vp = (tp * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    df["VWAP_cum"] = cum_vp / cum_vol

    # Heikin-Ashi
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    ha_open = ha_close.copy()
    if len(ha_open) > 0:
        ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2.0
    for i in range(1, len(ha_open)):
        ha_open.iat[i] = (ha_open.iat[i - 1] + ha_close.iat[i - 1]) / 2.0
    df["HA_Open"] = ha_open
    df["HA_Close"] = ha_close
    df["HA_High"] = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
    df["HA_Low"] = df[["Low", "HA_Open", "HA_Close"]].min(axis=1)

    # ADX (using DI sums approach)
    def _adx(df_, n=14):
        up_move = df_["High"].diff()
        down_move = -df_["Low"].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = _true_range(df_)
        atr = tr.rolling(n).mean()
        plus_dm_sm = pd.Series(plus_dm, index=df_.index).rolling(window=n).sum()
        minus_dm_sm = pd.Series(minus_dm, index=df_.index).rolling(window=n).sum()
        plus_di = 100 * (plus_dm_sm / atr)
        minus_di = 100 * (minus_dm_sm / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(n).mean()
        return plus_di, minus_di, adx

    df["+DI_14"], df["-DI_14"], df["ADX_14"] = _adx(df, 14)

    # Aroon (n=25)
    def _aroon(df_, n=25):
        # Aroon up/down in percentage. This implementation uses rolling apply.
        def single_aroon_up(arr):
            # arr is an array of highs in the window
            idx = np.argmax(arr)
            periods_since_high = (len(arr) - 1) - idx
            return ((n - periods_since_high) / n) * 100.0

        def single_aroon_down(arr):
            idx = np.argmin(arr)
            periods_since_low = (len(arr) - 1) - idx
            return ((n - periods_since_low) / n) * 100.0

        aroon_up = df_["High"].rolling(window=n).apply(single_aroon_up, raw=True)
        aroon_down = df_["Low"].rolling(window=n).apply(single_aroon_down, raw=True)
        return aroon_up, aroon_down

    try:
        df["Aroon_Up_25"], df["Aroon_Down_25"] = _aroon(df, 25)
    except Exception:
        df["Aroon_Up_25"] = np.nan
        df["Aroon_Down_25"] = np.nan

    # Vortex Indicator
    def _vortex(df_, n=14):
        tr = _true_range(df_)
        trn = tr.rolling(n).sum()
        vmp = (df_["High"] - df_["Low"].shift(1)).abs().rolling(n).sum()
        vmm = (df_["Low"] - df_["High"].shift(1)).abs().rolling(n).sum()
        vip = vmp / trn
        vim = vmm / trn
        return vip, vim

    df["Vortex_Pos_14"], df["Vortex_Neg_14"] = _vortex(df, 14)

    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper that runs all modular functions in a safe order.
    """
    df = df.copy()
    df = add_daily_return(df)
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volume_indicators(df)
    df = add_volatility_indicators(df)
    df = add_hybrid_indicators(df)
    # cleanup infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

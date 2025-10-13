import pandas as pd


def indexify_date(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.set_index("Date")
    df = df.sort_index()
    return df


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    df["Daily_Return"] = df["Close"].pct_change()
    return df

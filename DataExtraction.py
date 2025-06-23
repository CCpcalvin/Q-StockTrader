import yfinance as yf
import pandas as pd
import numpy as np
import os, shutil

# Folder to store the data
DATA_PATH = "data"
TICKER_PATH = "tickers.csv"


def getStockPool():
    df = pd.read_csv(TICKER_PATH)
    return df


def getData(ticker: str):
    data: pd.DataFrame = yf.download(ticker, "2013-10-01", "2023-10-01")
    data.to_csv(os.path.join(DATA_PATH, f"{ticker}.csv"))


def getCurrentDataPool():
    fileList = os.listdir(DATA_PATH)
    return set(os.path.splitext(file)[0] for file in fileList)


def getDataPool(reinstall: bool = False):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # Get the stock pool
    ticker_df = getStockPool()

    # Get the current historical data pool 
    currentDataPool = getCurrentDataPool()

    # If reinstall is true, then we always get the data
    # Else, download the history price data if it is not in the data pool
    for _, row in ticker_df.iterrows():
        if reinstall or row["symbol"] not in currentDataPool:
            if row["symbol"] is not np.nan:
                print(f"Getting {row['symbol']}")
                getData(row["symbol"])


def cleanDataPool():
    shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH)


# getStockPool()
# getData("AAPL")
getDataPool()
# cleanDataPool()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import os, random


RAW_DATA_PATH = "data"
FORMAT_DATA_PATH = os.path.join("formatted_data", "deep q-learning")


def getSMA(df: pd.DataFrame, period: int):
    # Calculate moving average
    df[f"SMA with period {period}"] = df["Adj Close"].rolling(period).mean()

    return df


def getSMATest(df: pd.DataFrame):
    getSMA(df, 5)
    getSMA(df, 150)

    print(df)

    # Set figure size
    plt.figure(figsize=(12, 9))

    # Plot the resistance and support level
    plt.plot(df["Adj Close"], label="Adj Close")
    plt.plot(df["SMA with period 5"], label="SMA with period 5")
    plt.plot(df["SMA with period 150"], label="SMA with period 150")

    # Plot buy and sell signal
    plt.legend()
    plt.show()

    #! Test 
    plt.figure(figsize=(12, 9))

    # Plot the resistance and support level
    plt.plot(df["Adj Close"] / df["Adj Close"], label="Adj Close")
    plt.plot(df["SMA with period 5"] / df["Adj Close"], label="SMA with period 5")
    plt.plot(df["SMA with period 150"] / df["Adj Close"], label="SMA with period 150")

    # Plot buy and sell signal
    plt.legend()
    plt.show()


def getResistance(df: pd.DataFrame, order: int = 5):
    # Found local maximum
    maxIdx = sg.argrelmax(df["Adj Close"].values, order=order)[0]

    if len(maxIdx) == 0:
        df["Resistance"] = np.nan
        return df

    # Generate index array based on the local extremum index array
    DataNum = df["Adj Close"].size
    # Get the resistance level index array
    resistanceIdx = np.zeros(DataNum, dtype=int)
    for idx in range(1, len(maxIdx)):
        previousMaxIdx = maxIdx[idx - 1]
        currentMaxIdx = maxIdx[idx]
        resistanceIdx[previousMaxIdx + order : currentMaxIdx + order] = previousMaxIdx

    # Fix the end index
    resistanceIdx[maxIdx[-1] + order :] = maxIdx[-1]

    # Extract the resistance and support level data based on the index array
    df["Resistance"] = df.iloc[resistanceIdx]["Adj Close"].to_list()

    # Fix the beginning of the data
    # Since we do not know the resistance and support level at the beginning,
    # Replace the data with NaN
    df.iloc[: resistanceIdx[0] + order, df.columns.get_loc("Resistance")] = np.nan

    return df


def getSupport(df: pd.DataFrame, order: int = 5):
    # Found local minimum
    minIdx = sg.argrelmin(df["Adj Close"].values, order=order)[0]

    if len(minIdx) == 0:
        df["Support"] = np.nan
        return df

    # Generate index array based on the local extremum index array
    DataNum = df["Adj Close"].size

    # Get the support level index array
    supportIdx = np.zeros(DataNum, dtype=int)
    for idx in range(1, len(minIdx)):
        previousMinIdx = minIdx[idx - 1]
        currentMinIdx = minIdx[idx]
        supportIdx[previousMinIdx + order : currentMinIdx + order] = previousMinIdx

    # Fix the end index
    supportIdx[minIdx[-1] + order :] = minIdx[-1]

    # Extract the resistance and support level data based on the index array
    df["Support"] = df.iloc[supportIdx]["Adj Close"].to_list()

    # Fix the beginning of the data
    # Since we do not know the resistance and support level at the beginning,
    # Replace the data with NaN
    df.iloc[: supportIdx[0] + order, df.columns.get_loc("Support")] = np.nan

    return df


# Get s_{RS}
def getRS(df: pd.DataFrame, order: int = 5):
    df = getResistance(df, order=order)
    df = getSupport(df, order=order)

    return df


# getSRS Test
def getRSTest(df: pd.DataFrame):
    df = getRS(df, order=10)
    print(df)

    # Set figure size
    plt.figure(figsize=(12, 9))

    # Plot the resistance and support level
    plt.plot(df["Adj Close"], label="Adj Close")
    plt.plot(df["Resistance"], label="Resistance")
    plt.plot(df["Support"], label="Support")

    # Plot buy and sell signal
    plt.legend()
    plt.show()

    #! Test
    # Try normalize the resistance and support by the adj close price
    # Set figure size
    # plt.figure(figsize=(12, 9))

    # # Plot the resistance and support level
    # plt.plot(df["Adj Close"] / df["Adj Close"], label="Adj Close")
    # plt.plot(df["Resistance"] / df["Adj Close"], label="Resistance")
    # plt.plot(df["Support"] / df["Adj Close"], label="Support")

    # # Plot buy and sell signal
    # plt.legend()
    # plt.show()


def getRSI(df: pd.DataFrame, period: int = 14):
    # Get the gain and loss
    PrevClose = df["Adj Close"].shift(1)
    CloseDiff = df["Adj Close"] - PrevClose
    df["Gain"] = CloseDiff.where(CloseDiff > 0)
    df["Loss"] = -1 * CloseDiff.where(CloseDiff < 0)

    # Calculate the average gain
    # Observe that the pd.Rolling will return NaN for any NaN value inside the Rolling
    # Set min_periods = 1 to undo this checking
    # (See https://stackoverflow.com/questions/40814201/pandas-rolling-gives-nan for more information)
    # But we have to eliminate the first `period - 1` rows of data by our own
    df["Average Gain"] = df["Gain"].rolling(period, min_periods=1).mean()
    df["Average Loss"] = df["Loss"].rolling(period, min_periods=1).mean()
    df.iloc[: period - 1, df.columns.get_loc("Average Gain")] = np.nan
    df.iloc[: period - 1, df.columns.get_loc("Average Loss")] = np.nan

    # Calculate the RSI value
    RS_data = df["Average Gain"] / df["Average Loss"]
    df["RSI"] = 100 - 100 / (1 + RS_data)

    return df


def getRSITest(df: pd.DataFrame):
    period = 14
    getRSI(df, period=period)

    print(df[period:])


def main():
    for file in os.listdir(RAW_DATA_PATH):
        print(f"Formatted {file}...")
        df = pd.read_csv(os.path.join(RAW_DATA_PATH, file))
        df = getSMA(df, 2)
        df = getSMA(df, 5)
        df = getSMA(df, 50)
        df = getSMA(df, 150)
        df = getSMA(df, 200)

        df = getRS(df)
        df = getRSI(df)
        df.to_csv(os.path.join(FORMAT_DATA_PATH, file), index=False)


if __name__ == "__main__":
    # main()

    file = random.choice(os.listdir(RAW_DATA_PATH))
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, file))
    getSMATest(df)
    # getRSTest(df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import os, random

RAW_DATA_PATH = "data"
TRAIN_DATA_PATH = os.path.join("formatted_data", "q-learning", "train")
TEST_DATA_PATH = os.path.join("formatted_data", "q-learning", "test")
STAT_DATA_PATH = TRAIN_DATA_PATH
STOCK_DATA_PATH = os.path.join(TRAIN_DATA_PATH, "stock")

S_MA_COL = "S_MA"
S_RS_COL = "S_RS"
S_RSI_COL = "S_RSI"

S_MA_NUM = 4
S_RS_NUM = 3
S_RSI_NUM = 4

FONTSIZE = 18
MARKERSIZE = 16
LINEWIDTH = 2
COLORMAP = ["tab:blue", "tab:orange", "tab:green", "tab:cyan", "tab:brown"]


# Get s_{MA} for entire DataFrame
def getSMA(df: pd.DataFrame, shortPeriod: int = 5, longPeriod: int = 150):
    # Calculate moving average
    df["ShortSMA"] = df["Adj Close"].rolling(shortPeriod).mean()
    df["LongSMA"] = df["Adj Close"].rolling(longPeriod).mean()

    # Get the previous moving average data
    PrevShortSMA = df["ShortSMA"].shift(1)
    PrevLongSMA = df["LongSMA"].shift(1)

    # Get s_{MA} state
    s_MA_eq_1 = (PrevShortSMA < PrevLongSMA) & (df["ShortSMA"] >= df["LongSMA"])
    s_MA_eq_2 = (PrevShortSMA > PrevLongSMA) & (df["ShortSMA"] <= df["LongSMA"])
    s_MA_eq_3 = (PrevShortSMA >= PrevLongSMA) & (df["ShortSMA"] >= df["LongSMA"])

    s_MA_eq_nan = PrevLongSMA.isna()
    df[S_MA_COL] = np.where(
        s_MA_eq_1,
        1,
        np.where(
            s_MA_eq_2, 2, np.where(s_MA_eq_3, 3, np.where(~s_MA_eq_nan, 4, np.nan))
        ),
    )

    return df


# getSMA Test
def getSMATest(df: pd.DataFrame):
    shortPeriod = 5
    longPeriod = 150
    df = getSMA(df, shortPeriod=shortPeriod, longPeriod=longPeriod)
    # Set figure size
    plt.figure(figsize=(12, 9))

    # Plot the adj close, short moving average and log moving average
    plt.plot(df["Adj Close"], label="Adj Close", linewidth=LINEWIDTH, color=COLORMAP[0])
    plt.plot(
        df["ShortSMA"],
        label=f"{shortPeriod} SMA",
        linewidth=LINEWIDTH,
        color=COLORMAP[1],
    )
    plt.plot(
        df["LongSMA"], label=f"{longPeriod} SMA", linewidth=LINEWIDTH, color=COLORMAP[2]
    )

    # Plot buy and sell signal
    BuyPrice = df["Adj Close"].where(df[S_MA_COL] == 1.0)
    SellPrice = df["Adj Close"].where(df[S_MA_COL] == 2.0)
    plt.plot(
        BuyPrice, "^", label="Buy Signal", markersize=MARKERSIZE, color=COLORMAP[3]
    )
    plt.plot(
        SellPrice, "v", label="Sell Signal", markersize=MARKERSIZE, color=COLORMAP[4]
    )

    plt.xlabel("Time Index", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.ylabel("Price", fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()


# Get s_{RS}
def getSRS(df: pd.DataFrame, order: int = 5):
    # Found local extremum
    maxIdx = sg.argrelmax(df["Adj Close"].values, order=order)[0]
    minIdx = sg.argrelmin(df["Adj Close"].values, order=order)[0]

    if len(maxIdx) == 0 or len(minIdx) == 0:
        df[S_RS_COL] = np.nan
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

    # Get the support level index array
    supportIdx = np.zeros(DataNum, dtype=int)
    for idx in range(1, len(minIdx)):
        previousMinIdx = minIdx[idx - 1]
        currentMinIdx = minIdx[idx]
        supportIdx[previousMinIdx + order : currentMinIdx + order] = previousMinIdx

    # Fix the end index
    supportIdx[minIdx[-1] + order :] = minIdx[-1]

    # Extract the resistance and support level data based on the index array
    df["Resistance"] = df.iloc[resistanceIdx]["Adj Close"].to_list()
    df["Support"] = df.iloc[supportIdx]["Adj Close"].to_list()

    # Fix the beginning of the data
    # Since we do not know the resistance and support level at the beginning,
    # Replace the data with NaN
    df.iloc[: resistanceIdx[0] + order, df.columns.get_loc("Resistance")] = np.nan
    df.iloc[: supportIdx[0] + order, df.columns.get_loc("Support")] = np.nan

    # Get s_{RS} state
    df[S_RS_COL] = np.where(
        df["Adj Close"] > df["Resistance"],
        1,
        np.where(
            df["Adj Close"] < df["Support"], 
            2,
            np.where(
                ~df["Resistance"].isna() & ~df["Support"].isna(), 3, np.nan
            )
        ),
    )

    return df


# getSRS Test
def getSRSTest(df: pd.DataFrame):
    df = getSRS(df, order=5)
    # Set figure size
    plt.figure(figsize=(12, 9))

    # Plot the resistance and support level
    plt.plot(df["Adj Close"], label="Adj Close", linewidth=LINEWIDTH, color=COLORMAP[0])
    plt.plot(
        df["Resistance"], label="Resistance", linewidth=LINEWIDTH, color=COLORMAP[1]
    )
    plt.plot(df["Support"], label="Support", linewidth=LINEWIDTH, color=COLORMAP[2])

    # Plot buy and sell signal
    prev_df = df.shift(1)
    BuyPrice = df["Adj Close"].where((df[S_RS_COL] == 1.0) & (prev_df[S_RS_COL] != 1.0))
    SellPrice = df["Adj Close"].where((df[S_RS_COL] == 2.0) & (prev_df[S_RS_COL] != 2.0))
    plt.plot(
        BuyPrice, "^", label="Buy Signal", markersize=MARKERSIZE, color=COLORMAP[3]
    )
    plt.plot(
        SellPrice, "v", label="Sell Signal", markersize=MARKERSIZE, color=COLORMAP[4]
    )

    plt.xlabel("Time Index", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.ylabel("Price", fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()


# Get s_{RAI}
def getSRSI(df: pd.DataFrame, period: int = 14, upperBound=70, lowerBound=30):
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

    # Get s_{RSI} state
    df[S_RSI_COL] = np.where(
        df["RSI"] < lowerBound,
        1,
        np.where(
            df["RSI"] > upperBound,
            2,
            np.where(
                (df["RSI"] >= lowerBound) & (df["RSI"] <= 50),
                3,
                np.where(~df["RSI"].isna(), 4, np.nan),
            ),
        ),
    )

    return df


def getSRSITest(df: pd.DataFrame):
    df = getSRSI(df)
    print(df.head(30))
    print(df[250:300])

    print(df[(df[S_RSI_COL] == 1.0)])
    print(df[(df[S_RSI_COL] == 2.0)])
    print(df[(df[S_RSI_COL] == 3.0)])
    print(df[(df[S_RSI_COL] == 4.0)])


def stateFolderFormat(s_MA: int, s_RS: int, s_RSI: int):
    return os.path.join(STAT_DATA_PATH, f"{s_MA}{s_RS}{s_RSI}")


def createFolder(folderPath: str):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)


def Format(train_test_split_ratio: float = 0.8):
    # Initialize the folder
    createFolder(STOCK_DATA_PATH)
    createFolder(TRAIN_DATA_PATH)
    createFolder(TEST_DATA_PATH)

    # Initialize the state-list pair
    # The list is to store all the dataframe
    # We combine it together at the end to achieve O(n) complexity
    StateDir: dict[tuple[int, int, int], list[pd.DataFrame]] = {}
    for s_MA in range(1, S_MA_NUM + 1):
        for s_RS in range(1, S_RS_NUM + 1):
            for s_RSI in range(1, S_RSI_NUM + 1):
                StateDir[(s_MA, s_RS, s_RSI)] = []

    # Loop over all data
    for file in os.listdir(RAW_DATA_PATH):
        # Print message
        print(f"Formatting {file}...")

        # Read a csv file
        df = pd.read_csv(os.path.join(RAW_DATA_PATH, file))

        # If we get the empty dataframe, skip it
        if df.shape[0] != 0:
            # Do analysis
            df = getSMA(df)
            df = getSRS(df)
            df = getSRSI(df)

            # Drop some rows where one of the state is na
            df = df[
                (~df[S_MA_COL].isna())
                & (~df[S_RS_COL].isna())
                & (~df[S_RSI_COL].isna())
            ]
            if df.shape[0] == 0:
                continue

            # Now the data are sorting ascending by time
            dataNum = len(df)
            split_idx = round(dataNum * train_test_split_ratio)
            train_df = df.iloc[:split_idx, :].copy()
            test_df = df.iloc[split_idx:, :]

            # Export the file
            train_df.to_csv(os.path.join(STOCK_DATA_PATH, file), index=False)
            test_df.to_csv(os.path.join(TEST_DATA_PATH, file), index=False)

            # Separate the data by state
            train_df["Symbol"] = file.replace(".csv", "")
            for s_MA in range(1, S_MA_NUM + 1):
                for s_RS in range(1, S_RS_NUM + 1):
                    for s_RSI in range(1, S_RSI_NUM + 1):
                        extracted_df = train_df[
                            (train_df[S_MA_COL] == s_MA)
                            & (train_df[S_RS_COL] == s_RS)
                            & (train_df[S_RSI_COL] == s_RSI)
                        ]

                        # If it is not empty dataframe, export it to csv
                        if extracted_df.shape[0] != 0:
                            StateDir[(s_MA, s_RS, s_RSI)].append(
                                extracted_df[["Date", "Symbol"]]
                            )

    # Combine all the dataframe and export it
    for s_MA in range(1, S_MA_NUM + 1):
        for s_RS in range(1, S_RS_NUM + 1):
            for s_RSI in range(1, S_RSI_NUM + 1):
                if len(StateDir[(s_MA, s_RS, s_RSI)]) == 0:
                    print((s_MA, s_RS, s_RSI))

                pd.concat(StateDir[(s_MA, s_RS, s_RSI)]).to_csv(
                    os.path.join(STAT_DATA_PATH, f"{s_MA}{s_RS}{s_RSI}.csv"),
                    index=False,
                )


if __name__ == "__main__":
    # Main Task
    Format(train_test_split_ratio=0.8)

    # Test
    # df = pd.read_csv(
    # os.path.join(RAW_DATA_PATH, random.choice(os.listdir(RAW_DATA_PATH)))
    # )
    # df = pd.read_csv(os.path.join(RAW_DATA_PATH, "AAPL.csv"))
    # getSMATest(df)
    # getSRSTest(df)
    # getSRSITest(df)
    # print(df)

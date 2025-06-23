import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os, random, json

from Environment import Environment
from QLearningAgent import QLearningAgent
from Agent import MAAgent, RSAgent, RSIAgent, AgentAction
from enum import Enum

from collections import namedtuple

FONTSIZE = 18
MARKERSIZE = 16
LINEWIDTH = 2
COLORMAP = ["tab:blue", "tab:orange", "tab:green", "tab:cyan", "tab:brown"]

SHORTSMAPERIOD = 5
LONGSMAPERIOD = 150

RSILOWERBOUND = 30
RSIIUPPERBOUND = 70

TESTDIR = os.path.join("tests", "q-learning")


class PlotMode(Enum):
    MA = 1
    RS = 2 
    RSI = 3


def RunTest():
    testEnv = Environment(os.path.join("formatted_data", "q-learning"))

    # Q-Learning Method
    Q = QLearningAgent(testEnv)
    Q.Train(tol=1e-5)
    Q.Compile(os.path.join(TESTDIR, "q-learning-compiled.csv"))
    Q.RunTest(os.path.join(TESTDIR, "q-learning"))

    # Other simple agent
    a = MAAgent(testEnv)
    a.RunTest(os.path.join(TESTDIR, "MAAgent"))

    b = RSAgent(testEnv)
    b.RunTest(os.path.join(TESTDIR, "RSAgent"))

    c = RSIAgent(testEnv)
    c.RunTest(os.path.join(TESTDIR, "RSIAgent"))


def RunSummary():
    testEnv = Environment(os.path.join("formatted_data", "q-learning"))
    DailyReturnData = []
    TotalSymbol = len(testEnv.symbols)

    for count, symbol in enumerate(testEnv.symbols):
        if count % 100:
            print(f"Loading Progress {count}/{TotalSymbol}..")

        df = testEnv.GetTest(symbol)
        df["Log Adj Close"] = np.log(df["Adj Close"])
        DailyReturnData.append(df["Log Adj Close"].shift(-1) - df["Log Adj Close"])
    
    All_df = pd.concat(DailyReturnData)
    print(All_df.describe())


def ReRunTest():
    testEnv = Environment(os.path.join("formatted_data", "q-learning"))

    # Q-Learning Method
    Q = QLearningAgent(testEnv)
    Q.Load(os.path.join(TESTDIR, "q-learning-compiled.csv"))
    Q.RunTest(os.path.join(TESTDIR, "q-learning"))

    # # Other simple agent
    a = MAAgent(testEnv)
    a.RunTest(os.path.join(TESTDIR, "MAAgent"))

    b = RSAgent(testEnv)
    b.RunTest(os.path.join(TESTDIR, "RSAgent"))
    b.RunTest(os.path.join("tests", "q-learning", "RSAgent"))

    c = RSIAgent(testEnv)
    c.RunTest(os.path.join(TESTDIR, "RSIAgent"))


def VisualizeSignal(dataDir: str, mode: PlotMode | None = None):
    # Pick a random data
    sampleFile = random.choice(os.listdir(dataDir))
    # sampleFile = "AAPL.csv"

    if mode == None:
        df = pd.read_csv(os.path.join(dataDir, sampleFile))

        df["Buy Price"] = df["Adj Close"].where(df["Action"] == 1)
        df["Sell Price"] = df["Adj Close"].where(df["Action"] == 2)

        # Now we plot the price data
        xData = range(len(df["Adj Close"]))
        plt.figure(figsize=(12, 9))
        plt.plot(xData, df["Adj Close"], label="Price")
        plt.plot(xData, df["Buy Price"], "^", label="Buy Price")
        plt.plot(xData, df["Sell Price"], "v", label="Sell Price")

        plt.legend(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.show()

    if mode == PlotMode.MA:
        df = pd.read_csv(os.path.join(dataDir, sampleFile))

        df["Buy Price"] = df["Adj Close"].where(df["Action"] == 1)
        df["Sell Price"] = df["Adj Close"].where(df["Action"] == 2)

        # Now we plot the price data
        xData = range(len(df["Adj Close"]))
        plt.figure(figsize=(12, 9))
        plt.plot(xData, df["Adj Close"], label="Price", linewidth=LINEWIDTH)
        plt.plot(xData, df["ShortSMA"], label=f"{SHORTSMAPERIOD} SMA", linewidth=LINEWIDTH)
        plt.plot(xData, df["LongSMA"], label=f"{LONGSMAPERIOD} SMA", linewidth=LINEWIDTH)
        plt.plot(xData, df["Buy Price"], "^", label="Buy Price", markersize=MARKERSIZE)
        plt.plot(xData, df["Sell Price"], "v", label="Sell Price", markersize=MARKERSIZE)

        plt.xlabel("Time Index", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.ylabel("Price", fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.show()

    if mode == PlotMode.RS:
        df = pd.read_csv(os.path.join(dataDir, sampleFile))

        df["Buy Price"] = df["Adj Close"].where(df["Action"] == 1)
        df["Sell Price"] = df["Adj Close"].where(df["Action"] == 2)

        # Now we plot the price data
        xData = range(len(df["Adj Close"]))
        plt.figure(figsize=(12, 9))
        plt.plot(xData, df["Adj Close"], label="Price", linewidth=LINEWIDTH)
        plt.plot(xData, df["Resistance"], label=f"Resistance", linewidth=LINEWIDTH)
        plt.plot(xData, df["Support"], label=f"Support", linewidth=LINEWIDTH)
        plt.plot(xData, df["Buy Price"], "^", label="Buy Price", markersize=MARKERSIZE)
        plt.plot(xData, df["Sell Price"], "v", label="Sell Price", markersize=MARKERSIZE)

        plt.xlabel("Time Index", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.ylabel("Price", fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.show()

    if mode == PlotMode.RSI:
        df = pd.read_csv(os.path.join(dataDir, sampleFile))

        df["Buy Price"] = df["Adj Close"].where(df["Action"] == 1)
        df["Sell Price"] = df["Adj Close"].where(df["Action"] == 2)

        # Now we plot the price data
        xData = range(len(df["Adj Close"]))
        fig = plt.figure(figsize=(12, 9))
        gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[2, 1])
        ax = gs.subplots(sharex=True)
        
        ax[0].plot(xData, df["Adj Close"], label="Price", linewidth=LINEWIDTH)
        ax[0].plot(xData, df["Buy Price"], "^", label="Buy Price", markersize=MARKERSIZE)
        ax[0].plot(xData, df["Sell Price"], "v", label="Sell Price", markersize=MARKERSIZE)
        ax[0].set_ylabel("Price", fontsize=FONTSIZE)
        ax[0].tick_params(axis="y", labelsize=FONTSIZE)
        ax[0].legend(fontsize=FONTSIZE)
        
        ax[1].plot(xData, df["RSI"], label="RSI", linewidth=LINEWIDTH)
        RSIUpperBoundData = [RSIIUPPERBOUND for _ in xData]
        RSILowerBoundData = [RSILOWERBOUND for _ in xData]
        ax[1].plot(xData, RSIUpperBoundData, "--")
        ax[1].plot(xData, RSILowerBoundData, "--")
        ax[1].set_xlabel("Time Index", fontsize=FONTSIZE)
        ax[1].set_ylabel("RSI", fontsize=FONTSIZE)
        ax[1].tick_params(axis="both", labelsize=FONTSIZE)

        plt.tight_layout()
        plt.show()


def BrockTest(dataPath: str):
    UnconditionalReturnList = []
    BuyReturnList = []
    SellReturnList = []

    for file in os.listdir(dataPath):
        # Read csv
        df = pd.read_csv(os.path.join(dataPath, file))

        # Get the daily return
        df["Log Adj Close"] = np.log(df["Adj Close"])
        df["Daily Return"] = df["Log Adj Close"].shift(-1) - df["Log Adj Close"]
        df = df.dropna()

        # Get the daily return for buy signal only
        buy_df = df[df["Action"] == AgentAction.Buy.value]
        sell_df = df[df["Action"] == AgentAction.Sell.value]

        if len(df["Daily Return"]) != 0:
            UnconditionalReturnList.append(df["Daily Return"])

        if len(buy_df) != 0:
            BuyReturnList.append(buy_df["Daily Return"])

        if len(sell_df) != 0:
            SellReturnList.append(sell_df["Daily Return"])

    UnconditionalReturn = pd.concat(UnconditionalReturnList, ignore_index=True)
    BuyReturn = pd.concat(BuyReturnList, ignore_index=True)
    SellReturn = pd.concat(SellReturnList, ignore_index=True)

    Buy_Unconditional_tTest = stats.ttest_ind(
        BuyReturn,
        UnconditionalReturn,
        equal_var=False,
        alternative="greater",
    )

    Sell_Unconditional_tTest = stats.ttest_ind(
        SellReturn,
        UnconditionalReturn,
        equal_var=False,
        alternative="less",
    )

    Buy_Sell_tTest = stats.ttest_ind(
        BuyReturn,
        SellReturn,
        equal_var=False,
        alternative="greater",
    )

    print(f"Number of unconditional data: {len(UnconditionalReturn)}")
    print(f"Buy-Unconditional tTest: {Buy_Unconditional_tTest}")
    print(f"Sell-Unconditional tTest: {Sell_Unconditional_tTest}")
    print(f"Buy-Sell tTest: {Buy_Sell_tTest}")
    print("Description on buys")
    print(BuyReturn.describe())
    print("Description on sells")
    print(SellReturn.describe())

    return BuyReturn, SellReturn


def RunBrockTest():
    print("For MAAgent: ")
    MABuyReturn, MASellReturn = BrockTest(os.path.join(TESTDIR, "MAAgent", "test"))

    print("For RSAgent: ")
    RSBuyReturn, RSSellReturn = BrockTest(os.path.join(TESTDIR, "RSAgent", "test"))

    print("For RSIAgent: ")
    RSIBuyReturn, RSISellReturn = BrockTest(os.path.join(TESTDIR, "RSIAgent", "test"))

    print("For Q-learning Agent: ")
    QBuyReturn, QSellReturn = BrockTest(os.path.join(TESTDIR, "q-learning", "test"))

    # Now we compare buys and sells for each strategy
    data = namedtuple("data", ["Name", "Data"])
    Buys = [
        data("MA", MABuyReturn),
        data("RS", RSBuyReturn),
        data("RSI", RSIBuyReturn),
        data("Q", QBuyReturn),
    ]
    Sells = [
        data("MA", MASellReturn),
        data("RS", RSSellReturn),
        data("RSI", RSISellReturn),
        data("Q", QSellReturn),
    ]

    for sample1 in Buys:
        for sample2 in Buys:
            tTest = stats.ttest_ind(
                sample1.Data,
                sample2.Data,
                equal_var=False,
                alternative="greater",
            )
            print(f"The tTest on buys {sample1.Name}-{sample2.Name}: {tTest}")

    for sample1 in Sells:
        for sample2 in Sells:
            tTest = stats.ttest_ind(
                sample1.Data,
                sample2.Data,
                equal_var=False,
                alternative="less",
            )
            print(f"The tTest on sells {sample1.Name}-{sample2.Name}: {tTest}")


def mean(l: list):
    if len(l) == 0:
        return np.nan
    else:
        return sum(l) / len(l)


def ProfitTest():
    # Initialize variable
    dataPath: dict[str, str] = {
        "q-learning": os.path.join(TESTDIR, "q-learning", "test"),
        "MAAgent": os.path.join(TESTDIR, "MAAgent", "test"),
        "RSAgent": os.path.join(TESTDIR, "RSAgent", "test"),
        "RSIAgent": os.path.join(TESTDIR, "RSIAgent", "test"),
    }

    Profit: dict[str, list[float]] = {
        "q-learning": [],
        "MAAgent": [],
        "RSAgent": [],
        "RSIAgent": [],
    }

    # Read the file, record the profit as % increase of stock price - the transaction cost
    for AgentType in dataPath:
        Count = 1
        TotalFileCount = len(os.listdir(dataPath[AgentType]))

        for file in os.listdir(dataPath[AgentType]):
            print(
                f"Calculating the profile in {file} for {AgentType}. Progress: {Count} / {TotalFileCount}"
            )

            df = pd.read_csv(os.path.join(dataPath[AgentType], file))
            isHolding = False
            BuyPrice = 0.0

            for _, row in df.iterrows():
                if not isHolding and row["Action"] == AgentAction.Buy.value:
                    isHolding = True
                    BuyPrice = row["Adj Close"]

                elif isHolding and row["Action"] == AgentAction.Sell.value:
                    isHolding = False
                    Profit[AgentType].append(
                        # (row["Adj Close"] - BuyPrice - 2) / BuyPrice
                        (
                            row["Adj Close"]
                            - BuyPrice
                            - 0.001 * BuyPrice
                            - 0.001 * row["Adj Close"]
                        )
                        / BuyPrice
                    )

            Count += 1

    # Save the profit dictionary for later use
    with open("ProfitResult.json", "w") as f:
        json.dump(Profit, f)

    # Print the average profit for each method first
    print(f"The profit of q-learning is {mean(Profit['q-learning'])}")
    print(f"The profit of MAAgent is {mean(Profit['MAAgent'])}")
    print(f"The profit of RSAgent is {mean(Profit['RSAgent'])}")
    print(f"The profit of RSIAgent is {mean(Profit['RSIAgent'])}")

    # Do t-test analysis against q-learning
    q_learning_MA_tTest = stats.ttest_ind(
        Profit["q-learning"],
        Profit["MAAgent"],
        equal_var=False,
        alternative="greater",
    )
    print(f"tTest for q-learning against MA is: {q_learning_MA_tTest}")

    q_learning_RS_tTest = stats.ttest_ind(
        Profit["q-learning"],
        Profit["RSAgent"],
        equal_var=False,
        alternative="greater",
    )
    print(f"tTest for q-learning against RS is: {q_learning_RS_tTest}")

    q_learning_RSI_tTest = stats.ttest_ind(
        Profit["q-learning"],
        Profit["RSIAgent"],
        equal_var=False,
        alternative="greater",
    )
    print(f"tTest for q-learning against RSI is: {q_learning_RSI_tTest}")


if __name__ == "__main__":
    # RunTest()
    # RunSummary()
    # ReRunTest()
    # VisualizeSignal(os.path.join("tests", "q-learning", "MAAgent", "train"), PlotMode.MA)
    # VisualizeSignal(os.path.join("tests", "q-learning", "RSAgent", "train"), PlotMode.RS)
    # VisualizeSignal(os.path.join("tests", "q-learning1", "RSAgent", "train"), PlotMode.RS)
    # VisualizeSignal(os.path.join("tests", "q-learning", "RSIAgent", "train"), PlotMode.RSI)
    RunBrockTest()
    # ProfitTest()

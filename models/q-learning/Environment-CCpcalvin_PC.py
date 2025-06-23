import os, random
import pandas as pd


class EnvData:
    def __init__(self, data: pd.DataFrame, OriginData: pd.DataFrame) -> None:
        self.data = data
        self.OriginData = OriginData

    def __repr__(self) -> str:
        return f"(Sample: {self.data}, Origin: {self.OriginData})"


class Environment:
    def __init__(self, dataPath: str) -> None:
        # Get all the symbols
        self.dataPath = dataPath
        self.symbols = [
            os.path.splitext(file)[0]
            for file in os.listdir(os.path.join(dataPath, "test"))
        ]

        # Now set up the dictionary for storing the used data
        # We will load it dynamically
        self.train: dict[str, pd.DataFrame] = {}
        self.test: dict[str, pd.DataFrame] = {}

        # Load the state against sample group
        self.State2Sample: dict[tuple[int, int, int], pd.DataFrame] = {}
        self.AllState: list[tuple[int, int, int]] = []
        for s_MA in range(1, 5):
            for s_RS in range(1, 4):
                for s_RSI in range(1, 5):
                    if os.path.exists(os.path.join(dataPath, "train", f"{s_MA}{s_RS}{s_RSI}.csv")):
                        self.State2Sample[(s_MA, s_RS, s_RSI)] = pd.read_csv(
                            os.path.join(dataPath, "train", f"{s_MA}{s_RS}{s_RSI}.csv")
                        )

                        self.AllState.append((s_MA, s_RS, s_RSI))

    # Read the corresponding csv, extract only the useful col, then return it
    def LoadData(self, key: str):
        # Read the csv file
        df = pd.read_csv(os.path.join(self.dataPath, "stock", f"{key}.csv"))

        # Extract only useful column
        df = df[["Date", "Adj Close", "S_MA", "S_RS", "S_RSI"]]

        # Drop the na value and reindex
        return df.dropna().reset_index(drop=True)

    # Get Train dataframe
    def GetTrain(self, key: str):
        if key not in self.train:
            # Read the csv file
            df = pd.read_csv(
                os.path.join(self.dataPath, "train", "stock", f"{key}.csv")
            )

            # Extract only useful column
            df = df[
                [
                    "Date",
                    "Adj Close",
                    "S_MA",
                    "S_RS",
                    "S_RSI",
                    "ShortSMA",
                    "LongSMA",
                    "Resistance",
                    "Support",
                    "RSI",
                ]
            ]

            # Drop the na value and reindex
            self.train[key] = df.dropna().reset_index(drop=True)

        return self.train[key]

    # Get Test dataframe
    def GetTest(self, key: str):
        if key not in self.test:
            # Read the csv file
            df = pd.read_csv(os.path.join(self.dataPath, "test", f"{key}.csv"))

            # Extract only useful column
            df = df[
                [
                    "Date",
                    "Adj Close",
                    "S_MA",
                    "S_RS",
                    "S_RSI",
                    "ShortSMA",
                    "LongSMA",
                    "Resistance",
                    "Support",
                    "RSI",
                ]
            ]

            # Drop the na value and reindex
            self.test[key] = df.dropna().reset_index(drop=True)

        return self.test[key]

    def __len__(self):
        return len(self.symbols)

    # Get Random training sample
    # There are two sampling methods:
    # 1. sample uniformly the state
    # 2. sample the stock uniformly
    def GetRandomTrainingSample(self, epsilon: float = 0.8):
        sample = random.random()
        if sample > epsilon:
            # Random sample the state
            state = random.choice(self.AllState)
            df = self.State2Sample[state]

            # From a dataframe, random sample a data
            data = df.sample()
            symbol = str(data.iloc[0, 1])

            # Read the corresponding dataframe
            WholePriceData = self.GetTrain(symbol)
            PriceData = WholePriceData[WholePriceData["Date"] == data.iloc[0, 0]]

            return PriceData, WholePriceData

        else:
            # Random sample the stock
            stock = random.choice(self.symbols)
            WholePriceData = self.GetTrain(stock)
            PriceData = WholePriceData.sample()

            return PriceData, WholePriceData


def main():
    testEnv = Environment(os.path.join("formatted_data", "q-learning2"))
    print(testEnv.AllState)
    print(testEnv.GetTrain("AAPL"))
    print(testEnv.GetTest("AAPL"))

    for _ in range(10):
        print(testEnv.GetRandomTrainingSample())


if __name__ == "__main__":
    main()

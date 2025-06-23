import os, random
import pandas as pd


class Environment:
    def __init__(self, dataPath: str, train_test_ratio: float = 0.8) -> None:
        # Get all the symbols
        self.dataPath = dataPath
        self.symbols = [
            os.path.splitext(file)[0] for file in os.listdir(os.path.join(dataPath))
        ]

        # Shuffled the data, and split it into train data and test data
        random.shuffle(self.symbols)
        threshold = round(train_test_ratio * len(self.symbols))
        self.trainSymbols = self.symbols[:threshold]
        self.testSymbols = self.symbols[threshold:]

        # Now set up the dictionary for storing the used data
        # We will load it dynamically
        self.train = {}
        self.test = {}

        # Column list
        self.ColumnList = [
            "Date",
            "Adj Close",
            "Volume",
            "SMA with period 2",
            "SMA with period 5",
            "SMA with period 50",
            "SMA with period 150",
            "SMA with period 200",
            "Resistance",
            "Support",
            "RSI",
        ]

    # Read the corresponding csv, extract only the useful col, then return it
    def LoadData(self, key: str):
        # Read the csv file
        df = pd.read_csv(os.path.join(self.dataPath, f"{key}.csv"))

        # Extract only useful column
        df = df[
            [
                "Date",
                "Adj Close",
                "Volume",
                "SMA with period 2",
                "SMA with period 5",
                "SMA with period 50",
                "SMA with period 150",
                "SMA with period 200",
                "Resistance",
                "Support",
                "RSI",
            ]
        ]

        # Drop the na value and reindex
        return df.dropna().reset_index(drop=True)

    # Get the dataframe by key
    # If we haven't loaded it before, load it first and save it
    def __getitem__(self, key: str) -> pd.DataFrame:
        if key in self.trainSymbols:
            if key not in self.train:
                self.train[key] = self.LoadData(key)

            return self.train[key]

        elif key in self.testSymbols:
            if key not in self.test:
                self.test[key] = self.LoadData(key)

            return self.test[key]

        else:
            raise Exception(f"{key} not found in the {self.dataPath}")

    # Get Random training sample (dataframe) by simply random sample the symbol:
    def GetRandomTrainingSample(self):
        df = self[random.choice(self.trainSymbols)]
        row = df.sample()
        idx: int = row.index[0]
        return idx, df


def main():
    testEnv = Environment(os.path.join("formatted_data", "deep q-learning"))
    print(testEnv.trainSymbols)
    print()
    print(testEnv.testSymbols)

    print(testEnv.GetRandomTrainingSample())
    print(testEnv.GetRandomTrainingSample())
    print(testEnv.GetRandomTrainingSample())

    _, df = testEnv.GetRandomTrainingSample()
    print(df)
    print(df.shape[0])


if __name__ == "__main__":
    main()

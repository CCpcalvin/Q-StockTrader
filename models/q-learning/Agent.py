from __future__ import annotations
from Environment import Environment
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np
import os


class AgentAction(Enum):
    Buy = 1
    Sell = 2
    DoNth = 3


class State:
    def __init__(self, S_MA: float, S_RS: float, S_RSI: float, S_I: bool) -> None:
        self.S_MA = S_MA
        self.S_RS = S_RS
        self.S_RSI = S_RSI
        self.isHolding = S_I

    def __str__(self) -> str:
        return f"State(S_MA: {self.S_MA}, S_RS: {self.S_RS}, S_RSI: {self.S_RSI}, S_I: {self.isHolding})"

    def __repr__(self) -> str:
        return self.__str__()

    # Define the hash function in this way
    # Note that it gets the same hash function with other state with same attribute
    def __hash__(self) -> int:
        return (self.S_MA, self.S_RS, self.S_RSI, self.isHolding).__hash__()

    def __eq__(self, __value: State) -> bool:
        return (
            self.S_MA == __value.S_MA
            and self.S_RS == __value.S_RS
            and self.S_RSI == __value.S_RSI
            and self.isHolding == __value.isHolding
        )

    # Set the state based on 3 numbers
    def SetItemFromFloats(self, S_MA: float, S_RS: float, S_RSI: float, S_I: bool):
        self.S_MA = S_MA
        self.S_RS = S_RS
        self.S_RSI = S_RSI
        self.isHolding = S_I

    # Set the state based on a one-row dataframe
    # Assuming that the df.col = ["Date", "S_MA", "S_RS", "S_RSI"]
    def SetItemFromDF(self, df: pd.DataFrame):
        self.S_MA: float = df["S_MA"][0]
        self.S_RS: float = df["S_RS"][0]
        self.S_RSI: float = df["S_RSI"][0]

    # Set the state based on a pd.row object
    # Also assuming that the df.col = ["Date", "S_MA", "S_RS", "S_RSI"]
    def SetItemFromRow(self, row: pd.Series[float]):
        self.S_MA = row["S_MA"]
        self.S_RS = row["S_RS"]
        self.S_RSI = row["S_RSI"]

    def CopyState(self, toCopy: State):
        self.S_MA = toCopy.S_MA
        self.S_RS = toCopy.S_RS
        self.S_RSI = toCopy.S_RSI
        self.isHolding = toCopy.isHolding


class AbstractAgent(ABC):
    DefaultTestLocation = "tests"

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.CurrentState = State(0.0, 0.0, 0.0, False)

    def AvailableActions(self, state: State):
        if state.isHolding is False:
            return (AgentAction.Buy, AgentAction.DoNth)
        else:
            return (AgentAction.Sell, AgentAction.DoNth)

    def Buy(self):
        self.CurrentState.isHolding = True

    def Sell(self):
        self.CurrentState.isHolding = False

    @abstractmethod
    def Policy(self, state: State) -> AgentAction:
        pass

    # This function will run during the test phrase
    def _DoActionDuringTest(self, row: pd.Series) -> int:
        # Get the action based on the policy
        self.CurrentState.SetItemFromRow(row)
        action = self.Policy(self.CurrentState)

        # Do the corresponding action
        if action is AgentAction.Buy:
            self.Buy()
        elif action is AgentAction.Sell:
            self.Sell()

        # Return the action that the agent did for record
        return action.value

    # Automatically generate the testPath
    # The logic as follows, generate 1, 2, 3... directory in tests
    def _GetTestPath(self):
        CurrentTestNum = len(os.listdir(self.DefaultTestLocation)) + 1
        return os.path.join(self.DefaultTestLocation, f"{CurrentTestNum}")

    def _CreateDir(self, DirPath: str):
        if not os.path.exists(DirPath):
            os.makedirs(DirPath)

    def RunTest(self, testPath: Optional[str] = None):
        print("Start testing")

        # Get the testPath
        if testPath is None:
            testPath = self._GetTestPath()

        self._CreateDir(os.path.join(testPath, "train"))
        self._CreateDir(os.path.join(testPath, "test"))

        # Loop over all the data set 
        testNumber = len(self.env)
        for i, symbol in enumerate(self.env.symbols):
            print(f"Running test on {symbol}. Progression: {i + 1} / {testNumber}")

            # Now loop over the test dataset 
            # Reset the agent stock holding state
            self.CurrentState.isHolding = False

            # Get dataframe
            df = self.env.GetTest(symbol)
            df["Action"] = np.nan

            for idx, row in df.iterrows():
                df.at[idx, "Action"] = self._DoActionDuringTest(row)

            # Save the dataframe
            df.to_csv(os.path.join(testPath, "test", f"{symbol}.csv"), index=False)

            # Now loop over the train dataset 
            self.CurrentState.isHolding = False

            # Get dataframe
            df = self.env.GetTrain(symbol)
            df["Action"] = np.nan

            for idx, row in df.iterrows():
                df.at[idx, "Action"] = self._DoActionDuringTest(row)

            # Save the dataframe
            df.to_csv(os.path.join(testPath, "train", f"{symbol}.csv"), index=False)


        print("Finish testing.")
        print(f"The result are saved in {testPath}")


class MAAgent(AbstractAgent):
    def Policy(self, state: State) -> AgentAction:
        if state.S_MA == 1.0 and state.isHolding is False:
            return AgentAction.Buy

        elif state.S_MA == 2.0 and state.isHolding is True:
            return AgentAction.Sell

        else:
            return AgentAction.DoNth


class RSAgent(AbstractAgent):
    def Policy(self, state: State) -> AgentAction:
        if state.S_RS == 1.0 and state.isHolding is False:
            return AgentAction.Buy

        elif state.S_RS == 2.0 and state.isHolding is True:
            return AgentAction.Sell

        else:
            return AgentAction.DoNth


class RSIAgent(AbstractAgent):
    def Policy(self, state: State) -> AgentAction:
        if state.S_RSI == 1.0 and state.isHolding is False:
            return AgentAction.Buy

        elif state.S_RSI == 2.0 and state.isHolding is True:
            return AgentAction.Sell

        else:
            return AgentAction.DoNth


def main():
    testEnv = Environment("formatted_data")

    a = MAAgent(testEnv)
    a.RunTest()

    b = RSAgent(testEnv)
    b.RunTest()

    c = RSIAgent(testEnv)
    c.RunTest()


if __name__ == "__main__":
    main()

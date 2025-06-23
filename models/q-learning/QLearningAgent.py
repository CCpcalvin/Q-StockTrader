from Agent import AbstractAgent, State, AgentAction
from Environment import Environment
import pandas as pd
import math, random, copy, csv, os


class QLearningAgent(AbstractAgent):
    # Generate all possible states here
    # Note that
    # - s_MA = 1, 2, 3, 4
    # - s_RS = 1, 2, 3
    # - s_RSI = 1, 2, 3, 4
    # - s_I = 1, 2
    def GenerateAllPossibleState(self):
        AllPossibleStates: list[State] = []
        for s_MA in range(1, 5):
            for s_RS in range(1, 4):
                for s_RSI in range(1, 5):
                    for s_I in [True, False]:
                        AllPossibleStates.append(State(s_MA, s_RS, s_RSI, s_I))

        return AllPossibleStates

    def __init__(self, env: Environment) -> None:
        super().__init__(env)

        # Initialized the self.LastState for record
        self.LastState = State(0.0, 0.0, 0.0, False)

        # Initialized the Q-Table, LastQ-Table, Probability Table and Visit Table
        # Q-Table to store all the Q-value
        # LastQ-Table simply store the Q-value before a training loop
        # It is used to calculate the tolerance
        # The Probability Table just store the weighing (probability) of all actions based on Training Policy
        # Visit Table just store how many trials on that (state, action) pair
        self.QTable: dict[State, dict[AgentAction, float]] = {}
        self.LastQTable: dict[State, dict[AgentAction, float]] = {}
        self.PrTable: dict[State, dict[AgentAction, float]] = {}
        self.VisitsTable: dict[State, dict[AgentAction, int]] = {}

        for state in self.GenerateAllPossibleState():
            self.QTable[state] = {}
            self.LastQTable[state] = {}
            self.PrTable[state] = {}
            self.VisitsTable[state] = {}

            for action in self.AvailableActions(state):
                # Randomize the QTable
                # If we sett QTable to be 0 everywhere
                # It is possible that the training will stop early because error = 0
                self.QTable[state][action] = random.uniform(-0.1, 0.1)
                self.LastQTable[state][action] = self.QTable[state][action]
                self.PrTable[state][action] = 0.0
                self.VisitsTable[state][action] = 1

    # Initialized the training process
    def GetRandomSample(self):
        # Get random sample
        self.CurrentSample, self.CurrentDataFrame = self.env.GetRandomTrainingSample()
        self.DataLength = self.CurrentDataFrame.shape[0]
        self.CurrentIndex = self.CurrentSample.index[0]

        # Update the state
        self.CurrentState.S_MA = self.CurrentSample.at[self.CurrentIndex, "S_MA"]
        self.CurrentState.S_RS = self.CurrentSample.at[self.CurrentIndex, "S_RS"]
        self.CurrentState.S_RSI = self.CurrentSample.at[self.CurrentIndex, "S_RSI"]

    # Perform Boltzmann Policy during the training
    # Output a action based on the self.CurrentState
    def BoltzmannPolicy(self, temperature: float = 1.0):
        # Consider some special case first at the end of the daa
        if self.CurrentIndex + 1 == self.DataLength:
            # If you are holding a stock, while it is the last data from the dataframe
            # Then no matter what the agent should sell it
            if self.CurrentState.isHolding:
                return AgentAction.Sell

            # If you does not hold a stock, while it is the last daa from the dataframe
            # Then no matter what the agent should do nothing
            else:
                return AgentAction.DoNth

        # Now for the general rule
        QValues = self.QTable[self.CurrentState]
        PrValues = self.PrTable[self.CurrentState]

        # Update the probability table
        for action in QValues:
            PrValues[action] = math.exp(QValues[action] / temperature)

        # Random sample the action based on probability table
        return random.choices(list(PrValues.keys()), weights=list(PrValues.values()))[0]

    def TransactionCost(self, currentPrice: float):
        # return 1 / currentPrice
        return 0.001

    # Perform state transaction by updating self.CurrentState
    # Return the reward by doing `action` on self.CurrentState
    def StateTransaction(self, action: AgentAction):
        # Record the current state to self.LastState
        self.LastState.CopyState(self.CurrentState)

        # The case that the agent is not holding the stock
        if not self.CurrentState.isHolding:
            if action == AgentAction.Buy:
                # Get the profit
                currentPrice = float(
                    self.CurrentDataFrame.at[self.CurrentIndex, "Adj Close"]
                )
                futurePrice = float(
                    self.CurrentDataFrame.at[self.CurrentIndex + 1, "Adj Close"]
                )
                profit = (
                    futurePrice - currentPrice
                ) / currentPrice - self.TransactionCost(currentPrice)

                # Update the state
                self.CurrentIndex += 1
                self.CurrentSample = self.CurrentDataFrame.iloc[self.CurrentIndex]
                self.CurrentState.SetItemFromRow(self.CurrentSample)
                self.CurrentState.isHolding = True

                return profit

            else:
                self.GetRandomSample()
                return 0.0

        # The case that the agent is holding the stock
        else:
            if action == AgentAction.DoNth:
                # Get the profit
                currentPrice = float(
                    self.CurrentDataFrame.at[self.CurrentIndex, "Adj Close"]
                )
                futurePrice = float(
                    self.CurrentDataFrame.at[self.CurrentIndex + 1, "Adj Close"]
                )
                profit = (futurePrice - currentPrice) / currentPrice

                # Update the state
                self.CurrentIndex += 1
                self.CurrentSample = self.CurrentDataFrame.iloc[self.CurrentIndex]
                self.CurrentState.SetItemFromRow(self.CurrentSample)
                return profit

            else:
                currentPrice = float(
                    self.CurrentDataFrame.at[self.CurrentIndex, "Adj Close"]
                )
                TransactionCost = self.TransactionCost(currentPrice)
                self.CurrentState.isHolding = False
                self.GetRandomSample()
                return -TransactionCost

    # An updated version of error implementation
    # Here we calculate the mean error
    def CalculateError(self):
        Error = 0
        Count = 0
        # Ignore the data where LastQTable == QTable
        for state in self.LastQTable:
            for action in self.LastQTable[state]:
                if self.LastQTable[state][action] != self.QTable[state][action]:
                    Error += abs(
                        self.QTable[state][action] - self.LastQTable[state][action]
                    )
                    Count += 1

        return Error / Count

    # Define what a Training loop is
    def TrainLoop(self, gamma: float = 1.0):
        # Choose an action
        ChosenAction = self.BoltzmannPolicy()

        # Get the current Q-value estimation and state
        CurrentQValue = self.QTable[self.CurrentState][ChosenAction]

        # Update the current state and get the reward function
        # Also record the last state for updating Q-value
        Reward = self.StateTransaction(ChosenAction)

        # Get the new Q-value, update the LastQTable and QTable
        FutureReward = max(self.QTable[self.CurrentState].values())
        LearningRate = 1 / (1 + self.VisitsTable[self.LastState][ChosenAction])
        self.LastQTable[self.LastState][ChosenAction] = self.QTable[self.LastState][
            ChosenAction
        ]
        self.QTable[self.LastState][ChosenAction] = (
            1 - LearningRate
        ) * CurrentQValue + LearningRate * (Reward + gamma * FutureReward)

        # Update the visits table
        self.VisitsTable[self.LastState][ChosenAction] += 1

        # Now return the tolerance
        return self.CalculateError()
        # return abs(self.QTable[self.LastState][ChosenAction] - self.LastQTable[self.LastState][ChosenAction])

    # Now define the whole training process
    def Train(self, tol: float = 5e-4):
        # Initialize the agent
        self.GetRandomSample()
        Error = self.TrainLoop()

        # Loop over the self.TrainLoop until the error is smaller than the tolerance
        count = 0
        while Error > tol:
            Error = self.TrainLoop()
            count += 1
            if not count % 10:
                print(f"Finish training at {count} trial. The error is {Error}.")

    def Policy(self, state: State) -> AgentAction:
        QValues = self.QTable[state]
        return max(QValues, key=lambda k: QValues[k])

    # Here I only save the Q-Table 
    # The loaded Q agent should not be used for training again
    def Compile(self, outputCSV: str):
        df = pd.DataFrame(columns=["S_MA", "S_RS", "S_RSI", "S_I", "Action", "Q-Value"])
        for state in self.QTable:
            for action in self.QTable[state]:
                df.loc[len(df), :] = [
                    state.S_MA,
                    state.S_RS,
                    state.S_RSI,
                    state.isHolding,
                    action,
                    self.QTable[state][action],
                ]

        df.to_csv(outputCSV, index=False)
    
    def Str2Action(self, String: str):
        if String == "AgentAction.Buy":
            return AgentAction.Buy
        elif String == "AgentAction.Sell":
            return AgentAction.Sell
        elif String == "AgentAction.DoNth":
            return AgentAction.DoNth
        else:
            raise Exception(f"Unknown action {String}")
    
    def Str2Bool(self, String: str):
        if String == "True":
            return True
        elif String == "False":
            return False
        else:
            raise Exception(f"Cannot convert {String} to boolean")

    def Load(self, compiled_csv: str):
        with open(compiled_csv) as f:
            for row in csv.DictReader(f):
                print(row)
                state = State(
                    float(row["S_MA"]),
                    float(row["S_RS"]),
                    float(row["S_RSI"]),
                    self.Str2Bool(row["S_I"]),
                )
                action = self.Str2Action(row["Action"])
                self.QTable[state][action] = float(row["Q-Value"])


def StateTransactionTest(q: QLearningAgent):
    if q.CurrentState.isHolding is False:
        for action in [AgentAction.DoNth, AgentAction.Buy]:
            print(
                f"Now the agent perform {action} when isHolding is {q.CurrentState.isHolding}"
            )

            if action == AgentAction.DoNth:
                # Print action
                print(q.CurrentDataFrame)
                print(q.StateTransaction(action))
                print(q.CurrentDataFrame)

            else:
                print(q.CurrentDataFrame.loc[[q.CurrentIndex, q.CurrentIndex + 1]])
                print(q.CurrentSample)
                print(q.StateTransaction(action))
                print(q.CurrentSample)

    else:
        for action in [AgentAction.DoNth, AgentAction.Sell]:
            print(
                f"Now the agent perform {action} when isHolding is {q.CurrentState.isHolding}"
            )

            if action == AgentAction.DoNth:
                print(q.CurrentDataFrame.loc[[q.CurrentIndex, q.CurrentIndex + 1]])
                print(q.CurrentSample)
                print(q.StateTransaction(action))
                print(q.CurrentSample)

            else:
                print(q.CurrentDataFrame)
                print(q.StateTransaction(action))
                print(q.CurrentDataFrame)


def TrainPolicyTest(q: QLearningAgent):
    q.CurrentState = State(1.0, 1.0, 1.0, False)
    for action in q.QTable[q.CurrentState]:
        q.QTable[q.CurrentState][action] = 0.0
    print(q.BoltzmannPolicy())

    q.QTable[q.CurrentState][AgentAction.Buy] = 0.5
    q.QTable[q.CurrentState][AgentAction.DoNth] = -0.5
    print(q.BoltzmannPolicy())


def QTrainingLoopTest(q: QLearningAgent):
    q.GetRandomSample()
    LastQTable = copy.deepcopy(q.QTable)
    LastDataFrame = q.CurrentDataFrame
    LastIndex = q.CurrentIndex
    LastState = q.CurrentState

    q.TrainLoop()

    # Find the Q-Table difference
    for state in LastQTable:
        for action in LastQTable[state]:
            if LastQTable[state][action] != q.QTable[state][action]:
                print(state)
                print(action)
                print(q.QTable[state][action])
                print(LastIndex)
                print(q.CurrentIndex)


def LoadTest(complied_csv: str):
    testEnv = Environment(os.path.join("formatted_data", "q-learning"))
    Q = QLearningAgent(testEnv)
    Q.Load(complied_csv)
    print(Q.QTable)


def RunTest():
    testEnv = Environment(os.path.join("formatted_data", "q-learning"))
    Q = QLearningAgent(testEnv)
    Q.GetRandomSample()
    TrainPolicyTest(Q)


def main():
    testEnv = Environment("formatted_data")
    Q = QLearningAgent(testEnv)
    Q.Train()
    Q.Compile("compiled.csv")
    Q.RunTest()


if __name__ == "__main__":
    # main()
    # RunTest()
    LoadTest(os.path.join("tests", "q-learning", "q-learning-compiled1.csv"))

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from typing import NamedTuple

from Environment import Environment

import os, random, math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# Create a Transition class that store all the information about the Transition during q-learning
# Note that the state here is [
#   "Adj Close", "Volume", "SMAp=2", "SMAp=5", "SMAp=50", "SMAp=150", "SMAp=200",
#   "Resistance", "Support", "RSI", "Holdings", "Balance"
#   ]
class Transition(NamedTuple):
    State: torch.Tensor
    Action: int
    NextState: torch.Tensor
    Reward: float


# Create a buffer save all the Transition
class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define the DQN Net that agent need to use
# Define most of hyperparameter here
class DQN(nn.Module):
    def __init__(self, StateSpaceNum: int, ActionSpaceNum: int):
        super().__init__()

        # Define the architecture of the network
        self.layers = nn.Sequential(
            nn.Linear(StateSpaceNum, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, ActionSpaceNum),
        ).double()

    # A forward function for deep learning network
    def forward(self, x: torch.Tensor):
        return self.layers(x)

    # Copy another net to ths
    def copyNet(self, dqn: DQN):
        self.load_state_dict(dqn.state_dict())


class DQNAgent:
    def __init__(self, dataPath: str) -> None:
        # Create basic information for each agent for their transaction
        self.BalanceRange = (
            10000,
            50000,
        )  # Max amount that the agent allows for each stock
        self.MaxTransaction = (
            100  # Max number of share that you can buy per each transaction
        )

        # Start new episode after this number of trials
        # New episode means the agent will start over again with randomized balance
        self.TrialPerEpisode = 100

        # Create two DQN for agent
        # Note that the state here is [
        #   "Adj Close", "Volume", "SMAp=2", "SMAp=5", "SMAp=50", "SMAp=150", "SMAp=200",
        #   "Resistance", "Support", "RSI", "Holdings", "Balance"
        #   ]
        # So we got 12 inputs
        # The number of output will be integer between [-self.max_transaction, self.max_transaction]
        # indicating how many share the agent want to buy
        # In general we need two DQN
        # PolicyNet is the one we want to train
        # TargetNet is just the "previous estimation"
        self.State2IdxList = [
            "Adj Close",
            "Volume",
            "SMAp=2",
            "SMAp=5",
            "SMAp=50",
            "SMAp=150",
            "SMAp=200",
            "Resistance",
            "Support",
            "RSI",
            "Holdings",
            "Balance",
        ]
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PolicyNet = DQN(12, 2 * self.MaxTransaction + 1).to(self.Device)
        self.TargetNet = DQN(12, 2 * self.MaxTransaction + 1).to(self.Device)
        self.TargetNet.copyNet(self.PolicyNet)

        # To make use of batch optimization
        # The agent need to memory the last transition
        # So we can optimize the model faster by parallelization
        self.MemoryNum = 10000
        self.ReplayMemory = ReplayMemory(self.MemoryNum)

        # Define the deep-learning parameters
        self.LR = 5e-6  # Learning rate
        self.DecayRate = 0.99
        self.Optimizer = optim.AdamW(self.PolicyNet.parameters(), lr=self.LR)
        self.Scheduler = optim.lr_scheduler.ExponentialLR(
            self.Optimizer, gamma=self.DecayRate
        )
        # self.Loss_Fn = nn.SmoothL1Loss()
        # self.Loss_Fn = nn.MSELoss()
        self.Loss_Fn = nn.HuberLoss()
        self.BatchSize = 256

        # Define q-learning parameters
        self.ActionSpace = [
            i - self.MaxTransaction for i in range(2 * self.MaxTransaction + 1)
        ]
        self.Gamma = 0.99
        self.Tau = 0.5

        # Define the helper parameters specifically for our cases
        # Create an environment
        self.env = Environment(dataPath)

        # Start an new episode automatically (because I always forget during testing ;()
        self.InitEpisode()

    # Return the index given the column name
    def State2Idx(self, col: str):
        if col not in self.State2IdxList:
            raise IndexError

        return self.State2IdxList.index(col)

    # Return the value of each element given the 1*n 2-dimensional tensor
    def getElem(self, state: torch.Tensor, col: str):
        idx = self.State2Idx(col)
        return state[0, idx].item()

    # Input a state (1*n 2-dimensional torch.Tensor). Output an action which is integer
    # In general we implement epsilon-greedy training policy
    # Lastly we standard by desired action to real action that can really performed
    def TrainingPolicy(self, state: torch.Tensor, epsilon: float = 0.5):
        # If sample > epsilon, we pick the action leading to maximum Q-value
        # Otherwise we just random pick an action
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                idx: int = self.PolicyNet(state).max(1)[1].item()
                DesiredAction = self.ActionSpace[idx]
        else:
            DesiredAction = random.choice(self.ActionSpace)

        # If the agent do not have enough balance to buy enough shares
        # Then buy the maximum shares he can afford
        currentBalance = self.getElem(state, "Balance")
        currentPrice = self.getElem(state, "Adj Close")
        if DesiredAction * currentPrice > currentBalance:
            return math.floor(currentBalance / currentPrice)

        # If the agent does not have enough holding to sell the desired amount of shares
        # Then just sell all the shares he is holding
        currentHoldings = self.getElem(state, "Holdings")
        if -DesiredAction > currentHoldings:
            return int(-currentHoldings)

        # Otherwise, just do the desired action
        return DesiredAction

    def TransactionCost(self, Price: float):
        return 0.001 * Price

    # # Update the current state based on action
    # def StateTransition(self, action: int) -> Transition:
    #     # If the agent does not hold any share, and he decide not to buy it (action == self.Holdings == 0)
    #     # Or the agent decide to sell all of his sharing (action == - self.Holdings)
    #     # Then the agent want to leave this market and go to enter another market
    #     if action == -self.Holdings:
    #         # Update self.Holdings
    #         self.Holdings += action

    #         CurrentPrice = self.CurrentDataFrame.at[self.CurrentIndex, "Adj Close"]
    #         (
    #             self.CurrentIndex,
    #             self.CurrentDataFrame,
    #         ) = self.env.GetRandomTrainingSample()
    #         self.DataNum = self.CurrentDataFrame.shape[0]

    #         reward = -self.TransactionCost(-action * CurrentPrice)
    #         nextState = self.GetState(self.CurrentDataFrame, self.CurrentIndex)
    #         deepLearningAction = action + self.MaxTransaction
    #         transition = Transition(
    #             self.CurrentState, deepLearningAction, nextState, reward
    #         )

    #         self.CurrentState = nextState
    #         return transition

    #     # If the agent does not leave the market,
    #     # Then he will keep track of the same stock information
    #     # Update the self.CurrentIndex
    #     # Record the price and future price
    #     else:
    #         # Update self.Holdings
    #         self.Holdings += action

    #         # Get the future price
    #         CurrentPrice = self.CurrentDataFrame.at[self.CurrentIndex, "Adj Close"]
    #         self.CurrentIndex += 1
    #         FuturePrice = self.CurrentDataFrame.at[self.CurrentIndex, "Adj Close"]

    #         # Get reward, next state and pack all the information
    #         reward = self.Holdings * (
    #             FuturePrice - CurrentPrice
    #         ) - self.TransactionCost(abs(action) * CurrentPrice)
    #         nextState = self.GetState(self.CurrentDataFrame, self.CurrentIndex)
    #         deepLearningAction = action + self.MaxTransaction
    #         transition = Transition(
    #             self.CurrentState, deepLearningAction, nextState, reward
    #         )

    #         self.CurrentState = nextState
    #         return transition

    # Update the current state based on action
    def StateTransition(self, action: int) -> Transition:
        # Update self.Holdings
        self.Holdings += action

        # Get the future price
        CurrentPrice = self.CurrentDataFrame.at[self.CurrentIndex, "Adj Close"]
        self.CurrentIndex += 1
        FuturePrice = self.CurrentDataFrame.at[self.CurrentIndex, "Adj Close"]

        # Get reward, next state and pack all the information
        reward = self.Holdings * (FuturePrice - CurrentPrice) - self.TransactionCost(
            abs(action) * CurrentPrice
        )
        nextState = self.GetState(self.CurrentDataFrame, self.CurrentIndex)
        deepLearningAction = action + self.MaxTransaction
        transition = Transition(
            self.CurrentState, deepLearningAction, nextState, reward
        )

        self.CurrentState = nextState
        return transition

    # Now we update our model based on the ReplayMemory
    def OptimizeModel(self):
        # We only optimize our model if we have enough memory
        if len(self.ReplayMemory) >= self.BatchSize:
            # Sample trial in the memory, save it as batch
            transitions = self.ReplayMemory.sample(self.BatchSize)
            states, actions, next_states, rewards = tuple(zip(*transitions))

            state_batch = torch.cat(states)
            action_batch = torch.tensor(actions, device=self.Device).unsqueeze(0)
            next_state_batch = torch.cat(next_states)
            reward_batch = torch.tensor(rewards, device=self.Device)

            # Now we calculate the goal of the Optimal Q function i.e. Bell's Equation
            with torch.no_grad():
                future_reward = self.TargetNet(next_state_batch).max(1)[0]

            expected_Q_value = reward_batch + self.Gamma * future_reward

            # Calculate current Q value
            current_Q_value = self.PolicyNet(state_batch).gather(1, action_batch)

            # Now calculate the loss and optimize the model
            loss = self.Loss_Fn(current_Q_value, expected_Q_value.unsqueeze(0))
            self.Optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping to prevent gradient explosion
            nn.utils.clip_grad.clip_grad_value_(self.PolicyNet.parameters(), 100)
            self.Optimizer.step()

            self.Scheduler.step()

            return loss

    # Return a torch.Tensor object representing the state
    def GetState(self, df: pd.DataFrame, index: int):
        return torch.tensor(
            np.hstack(
                (df.iloc[index, 1:].to_numpy(), [self.Holdings, self.Balance])
            ).astype(float),
            device=self.Device,
        ).unsqueeze(0)

    def InitEpisode(self):
        # Reset holding and balance
        self.Holdings = 0
        self.Balance = random.uniform(self.BalanceRange[0], self.BalanceRange[1])

        # Random generate stock price data
        self.CurrentIndex, self.CurrentDataFrame = self.env.GetRandomTrainingSample()
        self.CurrentState = self.GetState(self.CurrentDataFrame, self.CurrentIndex)
        self.DataNum = self.CurrentDataFrame.shape[0]

        # Keep track of how many trials in each episode
        self.CurrentTrialInEP = 0

    def EpisodeLoop(self):
        # Choose the action
        action = self.TrainingPolicy(self.CurrentState)

        # Do the state transaction, record the transition and store it to memory
        transition = self.StateTransition(action)
        self.ReplayMemory.push(transition)

        # Optimization on policy network
        loss = self.OptimizeModel()

        # Update the target network based on soft update
        target_net_state_dict = self.TargetNet.state_dict()
        policy_net_state_dict = self.PolicyNet.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.Tau + target_net_state_dict[key] * (1 - self.Tau)
        self.TargetNet.load_state_dict(target_net_state_dict)

        # Update the count
        self.CurrentTrial += 1
        self.CurrentTrialInEP += 1

        return loss

    def StartTraining(self, trial=10000, debug: bool = True):
        self.CurrentTrial = 0
        self.InitEpisode()

        # Loop over desired trial
        loss_data = []

        while self.CurrentTrial < trial:
            if (
                self.CurrentIndex == self.DataNum - 1
                or self.CurrentTrialInEP >= self.TrialPerEpisode
            ):
                self.InitEpisode()

            else:
                loss = self.EpisodeLoop()
                if not self.CurrentTrial % 10:
                    print(f"Current loss: {loss} at trial {self.CurrentTrial}")

                    if debug:
                        loss_data.append(loss)

        if debug:
            loss_data = [loss.item() for loss in loss_data if loss is not None]

            plt.figure(figsize=(12, 9))
            plt.plot(loss_data)
            plt.show()
    
    def SaveNet(self, savePath: str = os.path.join("tests", "deep_q_learning_model.pt")):
        torch.save(self.PolicyNet.state_dict(), savePath)


def TrainingPolicyTest():
    agent = DQNAgent(os.path.join("formatted_data", "deep q-learning"))
    state = torch.tensor(
        np.hstack((agent.env["AAPL"].iloc[0, 1:].to_numpy(), [50, 10000])).astype(float)
    ).to(agent.Device)

    print(state)
    actions = []
    for _ in range(50):
        actions.append(agent.TrainingPolicy(state))

    print(actions)


def StateTransitionTest():
    agent = DQNAgent(os.path.join("formatted_data", "deep q-learning"))
    agent.InitEpisode()
    print(agent.CurrentDataFrame.iloc[agent.CurrentIndex : agent.CurrentIndex + 5])

    print(agent.CurrentState)
    print(agent.StateTransition(100))
    print(agent.CurrentState)
    print(agent.StateTransition(0))
    print(agent.CurrentState)
    print(agent.StateTransition(-75))
    print(agent.CurrentState)
    print(agent.StateTransition(-25))
    print(agent.CurrentDataFrame.iloc[agent.CurrentIndex : agent.CurrentIndex + 5])
    print(agent.CurrentState)
    print(agent.StateTransition(0))
    print(agent.CurrentDataFrame.iloc[agent.CurrentIndex : agent.CurrentIndex + 5])
    print(agent.CurrentState)


def OptimizeModelTest():
    agent = DQNAgent(os.path.join("formatted_data", "deep q-learning"))
    agent.CurrentTrial = 0
    agent.InitEpisode()
    state = agent.CurrentState
    print(agent.PolicyNet(state))

    for _ in range(agent.BatchSize):
        agent.EpisodeLoop()

    print(agent.PolicyNet(state))


def TrainingTest():
    agent = DQNAgent(os.path.join("formatted_data", "deep q-learning"))
    agent.StartTraining(100000)
    agent.SaveNet()


def main():
    # TrainingPolicyTest()
    # StateTransitionTest()
    # OptimizeModelTest()
    TrainingTest()


if __name__ == "__main__":
    main()

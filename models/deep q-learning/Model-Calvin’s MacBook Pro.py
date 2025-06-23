import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the DQN Net that agent need to use 
class DQN(nn.Module):
    def __init__(self, state, action):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, action),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class DQNAgent:
    def __init__(self) -> None:
        # Create basic information for each agent for their transaction 
        self.balance = 5000 # Max amount that the agent allows for each stock 

        # Create two DQN for agent 
        # Note that the state here is [
        #   "Adj Close", "Volume", "SMAp=2", "SMAp=5", "SMAp=50", "SMAp=150", "SMAp=200", 
        #   "Resistance", "Support", "RSI", "isHolding"
        #   ]
        # So we got 12 inputs
        # The output should be decrypted to a single value integer n 
        # if n is positive, then we buy n shares, if n is negative, hen we sell n shares 
        # if n is zero, then we do not save it 
        # assuming n can be in [-2^{11} - 1, 2^{11} - 1], so we need 12 bit for it 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PolicyNet = DQN(12, 12).to(self.device)


# 10/19/2023

From https://www.investopedia.com/terms/s/sma.asp, 
  - Short: 50, Long: 200

# 10/22/2023

Copy `tickers.csv` from https://github.com/shilewenuw/get_all_tickers/blob/master/get_all_tickers/EU_tickers.csv


# 10/26/2023

- Finish implementing basic model 
- Found out a problems which is we cannot sample certain pair 
- The tolerance will be stuck at certain sample
- It is because we have unfair distribution among the states 
  - For example S_I = 1 is so rare than S_I = 3
  - When the agent decides (not holding, do nothing) or (holding, sell), 
  - A "fair" sampling method should adopt
  - such that all state can be sampled fairly 
  - Hence the agent can explore all the state 

- We also have to give up the dynamically load data approach as we need global population data 

# 10/30/2023

- Finish better sampling method 
- But the state transaction implying that (1112) are rarer than (1111)
- Maybe give up the state transaction 
- Sample the state randomly first 


# 11/10/2023

- Finish building deep Q-learning algorithm
- But the algorithm cannot converge 
- I think
  - Maybe the data itself not enough for convergence 
  - Data too extreme for example Volume is too large when compared to other data

import gym
from gym import spaces
import pandas as pd
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, trading_fee=0.001):
        super(TradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Long, 2: Short
        # Observation space: [balance, position, current_price, other_features...]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3+ len(df.columns),), dtype=np.float32)
        
        self.reset()
        
    def reset(self):
        # Reset the environment to its initial state
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.done = False
        self.portfolio_value = self.balance
        return self._get_observation() 
    
    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}
        
        # Get current price and execute trade
        current_price = self.df.iloc[self.current_step]['close']
        self._execute_trade(action, current_price)

        
        
        self.current_step += 1
        if self.current_step >= len(self.df) - 1: 
            self.done = True
        
        # Calculate reward
        reward = self._calculate_reward()

        return self._get_observation, reward, self.done, {}
    
    def _execute_trade(self, action, current_price):
        # Execute trade
        if action == 1:  # Long
            long_amount = self.balance / current_price
            cost = long_amount * current_price * (1+self.trading_fee)
            if cost <= self.balance:
                self.position += long_amount
                self.balance -= cost
        elif action == 2:  # Short
            if self.position > 0:
                short_amount = self.position * current_price * (1 - self.transaction_fee_percent)
                self.balance += short_amount
                self.position = 0
    
    def _calculate_reward(self, current_price):
        # Calculate the new portfolio value
        new_portfolio_value = self.balance + self.position * current_price

        # Calculate the reward as the percentage change in portfolio value
        reward = (new_portfolio_value - self.portfolio_value) / self.portfolio_value

        # Update the portfolio value for the next step
        self.portfolio_value = new_portfolio_value

        return reward
    
    def _get_observation(self):
        obs = self.df.iloc[self.current_step].values
        return np.concatenate([[self.balance, self.position, self.df.iloc[self.current_step],['close']], obs])
    
    
    
    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Position: {self.position}')
        print(f'Current price: {self.df.iloc[self.current_step]["close"]}')
        print(f'Portfolio value: {self.portfolio_value:.2f}')
        print('--------------------')

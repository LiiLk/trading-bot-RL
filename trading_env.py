import gym
from gym import spaces
import pandas as pd
import numpy as np

class BitcoinTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=200, trading_fee=0.001, risk_factor=0.1):
        super(BitcoinTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.risk_factor = risk_factor
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Long, 2: Short
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)  # Open, High, Low, Close, Volume
        self.max_steps = len(df)
        
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        return self._get_obs()
    
    def step(self, action):
        if self.current_step >= self.max_steps - 1:
            self.current_step = 0  # Reset to the beginning of the data
        
        prev_close = self.df.iloc[self.current_step]['Close']
        self.current_step += 1
        current_close = self.df.iloc[self.current_step]['Close']
        
        if action == 1:  # Long
            self.position = self.balance / current_close
            self.balance -= self.position * current_close * (1 + self.trading_fee)
        elif action == 2:  # Short
            self.position = -self.balance / current_close
            self.balance += abs(self.position) * current_close * (1 - self.trading_fee)
        
        self.balance += self.position * (current_close - prev_close)
        
        reward = (self.balance - self.initial_balance) / self.initial_balance
        reward -= self.risk_factor * abs(self.position) * abs(current_close - prev_close) / prev_close
        
        if self.balance <= 0:
            done = True
        elif self.balance >= 1000000:  # 1 million target
            done = True
            reward += 1  # Additional reward for reaching target
        else:
            done = False
            
        info = {'balance': self.balance}
            
        return self._get_obs(), reward, done, info
    

    
    def _get_obs(self):
        obs = np.array([
            self.df.iloc[self.current_step]['Open'],
            self.df.iloc[self.current_step]['High'],
            self.df.iloc[self.current_step]['Low'],
            self.df.iloc[self.current_step]['Close'],
            self.df.iloc[self.current_step]['Volume']
        ], dtype=np.float32)
        return obs
import gym
from gym import spaces
import pandas as pd
import numpy as np
import logging
# Set up logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, csv_path, initial_balance=10000, leverage=100, risk_per_trade=0.01):
        super(TradingEnv, self).__init__()

        # Read the EURUSD data in one hour
        self.df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        self.df.set_index('timestamp', inplace=True)

        self.initial_balance = initial_balance
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade

        self.position = 0
        self.current_step = 0
        
        # Action space : 0 (hold), 1 (long), 2 (short)
        self.action_space = spaces.Discrete(3)

        # Define the observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
    
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        return self._next_observation()
    
    def _next_observation(self):
        current_data = self.df.iloc[self.current_step]
        print(self.df.iloc[1])
        obs = np.array([
            current_data['open'],
            current_data['high'],
            current_data['low'],
            current_data['close'],
            current_data['volume'],
            self.balance,
            self.position
        ])

        return obs
    
    def step(self, action):

        self._take_action(action)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        obs = self._next_observation()
        reward = self._calculate_reward()

        return obs, reward, done, {}
    
    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['close']

        if action == 1:  # Long
            if self.position <= 0:
                self._close_position(current_price)
                self._open_position(current_price, 1)
        elif action == 2:  # Short
            if self.position >= 0:
                self._close_position(current_price)
                self._open_position(current_price, -1)
    
    def _open_position(self, price, direction):
        position_size = self._calculate_position_size(price)
        self.position = position_size * direction
        self.entry_price = price
    
    def _close_position(self, price):
        if self.position != 0:
            profit = (price - self.entry_price) * self.position
            self.balance += profit
            self.position = 0
            self.entry_price = 0

    def _calculate_position_size(self, price):
        account_risk = self.balance * self.risk_per_trade
        pip_value = 0.0001 # Assuming 4 decimal places for EURUSD
        stop_loss_pips = 20 # For now i will do a fix stop loss 20 pips
        position_size = (account_risk / (stop_loss_pips * pip_value)) / price * self.leverage
        return position_size

        
    def _calculate_reward(self):
        current_price = self.df.iloc[self.current_step]['close']
        if self.position != 0:
            unrealized_profit = (current_price - self.entry_price) * self.position
            return unrealized_profit / self.initial_balance
        return 0

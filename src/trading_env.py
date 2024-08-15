import gym
from gym import spaces
import pandas as pd
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001, max_position=100):
        super(TradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.max_position = max_position

        # ATR parameters 
        self.atr_period = 14
        self.atr_multiplier = 2
        self.df['atr'] = self.calculate_atr(self.atr_period) 

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 0: Hold, 1: Long, 2: Short, 3: Close position
        # Observation space: [balance, position, current_price, other_features...]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + len(df.columns),), dtype=np.float32)
        self.reset()
        
    def reset(self):
        # Reset the environment to its initial state
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.done = False
        self.portfolio_value = self.balance
        self.entry_price = 0
        self.stop_loss_price = 0
        return self._get_observation() 
    
    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}
        
        # Get current price and execute trade
        current_price = self.df.iloc[self.current_step]['close']
        # Check for stop loss
        if self.position != 0:
            if (self.position > 0 and current_price <= self.stop_loss_price) or \
            (self.position < 0 and current_price >= self.stop_loss_price):
                self._close_position(current_price)

        self._execute_trade(action, current_price)
        
        # Update stop loss after each step
        if self.position != 0:
            self._set_dynamic_stop_loss(current_price)

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        reward = self._calculate_reward(current_price)

        return self._get_observation(), reward, self.done, {}
    
    def _execute_trade(self, action, current_price):
        # Execute trade
        if action == 1:  # Buy/Long
            if self.position <= 0:  # If short or no position, close it first
                self.balance += self.position * current_price * (1 - self.transaction_fee_percent)
                self.position = 0
            units_to_buy = min(self.max_position - self.position, 
                               self.balance // (current_price * (1 + self.transaction_fee_percent)))
            cost = units_to_buy * current_price * (1 + self.transaction_fee_percent)
            self.balance -= cost
            self.position += units_to_buy
            if self.position > 0:
                self.entry_price = current_price
                self._set_dynamic_stop_loss(current_price)
        elif action == 2:  # Sell/Short
            if self.position >= 0:
                self._close_position(current_price)
            units_to_short = min(self.max_position + self.position, 
                                self.balance // (current_price * (1 + self.transaction_fee_percent)))
            proceeds = units_to_short * current_price * (1 - self.transaction_fee_percent)
            self.balance += proceeds
            self.position -= units_to_short
            if self.position < 0:
                self.entry_price = current_price
                self._set_dynamic_stop_loss(current_price)

        elif action == 3:  # Close Position
            self.balance += self.position * current_price * (1 - self.transaction_fee_percent)
            self.position = 0
    
    def _set_dynamic_stop_loss(self, current_price):
        atr = self.df.iloc[self.current_step]['atr']
        if self.position > 0:  # Long position
            self.stop_loss_price = current_price - (atr * self.atr_multiplier)
        elif self.position < 0:  # Short position
            self.stop_loss_price = current_price + (atr * self.atr_multiplier)        
    def calculate_atr(self, period): 
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def _close_position(self, current_price):
        if self.position != 0:
            self.balance += self.position * current_price * (1 - self.trading_fee)
            self.position = 0
            self.entry_price = 0
            self.stop_loss_price = 0

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
        return np.concatenate([[self.balance, self.position, self.df.iloc[self.current_step]['close'], self.df.iloc[self.current_step]['atr']],obs])
      
    
    
    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Position: {self.position}')
        print(f'Current price: {self.df.iloc[self.current_step]["close"]}')
        print(f'Portfolio value: {self.portfolio_value:.2f}')
        print('--------------------')

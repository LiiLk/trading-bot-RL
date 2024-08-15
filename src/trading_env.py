import gym
from gym import spaces
import pandas as pd
import numpy as np
import logging
# Set up logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, trading_fee=0.001, risk_per_trade=0.01, leverage=1):
        super(TradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.cumulative_return = 0
        self.trades_executed = 0

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

        old_portfolio_value = self.balance + self.position * current_price

        
        # Check for stop loss before executing new trade
        if self.position != 0:
            if (self.position > 0 and current_price <= self.stop_loss_price) or \
               (self.position < 0 and current_price >= self.stop_loss_price):
                self._close_position(current_price)
        # Update stop loss after each step
            
        self._execute_trade(action, current_price)

        if self.position != 0:
            self._set_dynamic_stop_loss(current_price)
        
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        new_portfolio_value = self.balance + self.position * current_price
        reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        self.cumulative_return = (new_portfolio_value - self.initial_balance) / self.initial_balance

        logger.info(f"Step {self.current_step}: Action: {action}, Price: {current_price:.4f}, "
                    f"Balance: {self.balance:.2f}, Position: {self.position:.2f}, "
                    f"Reward: {reward:.6f}, Cumulative Return: {self.cumulative_return:.6f}")

        return self._get_observation(), reward, self.done, {
            'portfolio_value': new_portfolio_value,
            'cumulative_return': self.cumulative_return,
            'trades_executed': self.trades_executed
        }
    
    def _execute_trade(self, action, current_price):
        risk_amount = self.balance * self.risk_per_trade  # 1% risk per trade
        atr = self.df.iloc[self.current_step]['atr']

        if action == 1:  # Buy/Long
            if self.position <= 0:
                self._close_position(current_price)
            stop_loss = current_price - (atr * self.atr_multiplier)
            price_difference = current_price - stop_loss
            position_size = risk_amount / price_difference
            leveraged_position_size = position_size * self.leverage
            cost = leveraged_position_size * current_price * (1 + self.trading_fee)
            
            if cost <= self.balance * self.leverage:
                self.balance -= cost / self.leverage  # Only deduct the actual capital used
                self.position += leveraged_position_size
                self.entry_price = current_price
                self.stop_loss_price = stop_loss
                self.trades_executed += 1
                logger.info(f"Opened Long position: Size={leveraged_position_size:.2f}, Entry={current_price:.4f}, Stop={stop_loss:.4f}")

        elif action == 2:  # Sell/Short
            if self.position >= 0:
                self._close_position(current_price)
            stop_loss = current_price + (atr * self.atr_multiplier)
            price_difference = stop_loss - current_price
            position_size = risk_amount / price_difference
            leveraged_position_size = position_size * self.leverage
            proceeds = leveraged_position_size * current_price * (1 - self.trading_fee)
            
            if proceeds <= self.balance * self.leverage:
                self.balance += proceeds / self.leverage  # Only add the actual capital used
                self.position -= leveraged_position_size
                self.entry_price = current_price
                self.stop_loss_price = stop_loss
                self.trades_executed += 1
                logger.info(f"Opened Short position: Size={leveraged_position_size:.2f}, Entry={current_price:.4f}, Stop={stop_loss:.4f}")

        elif action == 3:  # Close Position
            self._close_position(current_price)
    
    def _set_dynamic_stop_loss(self, current_price):
        atr = self.df.iloc[self.current_step]['atr']
        if self.position > 0:  # Long position
            self.stop_loss_price = max(self.stop_loss_price, current_price - (atr * self.atr_multiplier))
        elif self.position < 0:  # Short position
            self.stop_loss_price = min(self.stop_loss_price, current_price + (atr * self.atr_multiplier))

    def calculate_atr(self, period): 
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def _close_position(self, current_price):
        if self.position != 0:
            pnl = (current_price - self.entry_price) * self.position
            self.balance += pnl / self.leverage  # Adjust PnL for leverage
            logger.info(f"Closed position: PnL={pnl:.2f}, Exit={current_price:.4f}")
            self.position = 0
            self.entry_price = 0
            self.stop_loss_price = 0

    
    def _get_observation(self):
        obs = self.df.iloc[self.current_step].values
        return np.concatenate([[self.balance, self.position, self.df.iloc[self.current_step]['close'], self.df.iloc[self.current_step]['atr']], obs])
    
    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Position: {self.position}')
        print(f'Current price: {self.df.iloc[self.current_step]["close"]}')
        print(f'Portfolio value: {self.portfolio_value:.2f}')
        print('--------------------')

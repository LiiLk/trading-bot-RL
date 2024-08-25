import gym
from gym import spaces
import pandas as pd
import numpy as np
import logging
from numba import jit

# Set up logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@jit(nopython=True)
def calculate_position_size(balance, risk_per_trade, price, stop_loss_distance, leverage):
    risk_amount = balance * risk_per_trade
    position_size = risk_amount / abs(stop_loss_distance)
    max_position_size = (balance * leverage) / price
    return min(position_size, max_position_size)

class TradingEnv(gym.Env):
    def __init__(self, csv_path, initial_balance=10000, leverage=100, risk_per_trade=0.01, atr_period=14, log_frequency=1000):
        super(TradingEnv, self).__init__()

        # Read the EURUSD data
        self.df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        self.df.set_index('timestamp', inplace=True)
        self.df['SMA_fast'] = self.df['close'].rolling(window=10).mean()
        self.df['SMA_slow'] = self.df['close'].rolling(window=30).mean()
        self.atr_period = atr_period  # Add this line to set the atr_period attribute
        self.atr = self._calculate_atr()
        
        # Prétraiter les observations
        self.observations = self._preprocess_observations()
        
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.transaction_fee_per_lot = 3 # 3$ per lot  

        self.position = 0
        self.current_step = 0

        self.stop_loss = None
        self.take_profit = None

        self.max_drawdown = 0.10 * initial_balance #10% max drawdown
        self.max_daily_drawdown = 0.05
        self.highest_balance = initial_balance
        self.daily_starting_balance = initial_balance
        self.last_day = None
        
        # Action space : 0 (hold), 1 (long), 2 (short)
        self.action_space = spaces.Discrete(3)

        # Define the observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    
        self.reset()

        self.log_frequency = log_frequency
        self.step_count = 0

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.highest_balance = self.initial_balance
        self.daily_starting_balance = self.initial_balance
        self.last_day = None
        return self._next_observation()
    
    def _preprocess_observations(self):
        return np.array([
            self.df['open'].values,
            self.df['high'].values,
            self.df['low'].values,
            self.df['close'].values,
            self.df['volume'].values,
            self.df['SMA_fast'].values,
            self.df['SMA_slow'].values,
        ])

    def _next_observation(self):
        obs = np.array([
            *self.observations[:, self.current_step],
            self.balance,
            self.position,
            self.entry_price if self.position != 0 else 0
        ])
        return obs
    
    def step(self, action):

        self._take_action(action)


        current_price = self.df.iloc[self.current_step]['close']

        if self.position > 0:
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                self._close_position(current_price)
        elif self.position < 0:
            if current_price >= self.stop_loss or current_price <= self.take_profit:
                self._close_position(current_price)

        if self._check_drawdown():
            done= True
            reward = -1 
        else: 
            self.current_step += 1
            done = self.current_step >= len(self.df) - 1
            reward = self._calculate_reward()

        obs = self._next_observation()
        
        self._update_daily_balance()

        self.step_count += 1
        if self.step_count % self.log_frequency == 0:
            logger.info(f"Step {self.current_step}, Balance: {self.balance}")

        # Check for NaN balance
        if np.isnan(self.balance):
            logger.error(f"Balance became NaN at step {self.current_step}")
            done = True

        return obs, reward, done, {}

    def _check_drawdown(self):
        if self.initial_balance - self.balance > self.max_drawdown:
            return True
        
        if self.daily_starting_balance - self.balance > self.max_daily_drawdown * self.daily_starting_balance:
            return True
        
        return False
    
    def _update_daily_balance(self):
        current_day = self.df.index[self.current_step].date()
        if self.last_day is None or current_day != self.last_day:
            self.daily_starting_balance = self.balance
            self.last_day = current_day
        
        self.highest_balance = max(self.highest_balance, self.balance)
    
    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['close']

        if self.position == 0:
            if action == 1:  # Buy
                self._open_position(current_price, 1)
            elif action == 2:  # Sell
                self._open_position(current_price, -1)
        else:
            if (action == 2 and self.position > 0) or (action == 1 and self.position < 0):
                self._close_position(current_price)

    def _calculate_atr(self):
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        close_shifted = np.concatenate(([close[0]], close[:-1]))  # Décalage de 1 pour close
        tr = np.maximum(high - low, np.abs(high - close_shifted), np.abs(low - close_shifted))
        atr = np.convolve(tr, np.ones(self.atr_period), 'valid') / self.atr_period
        atr = np.concatenate((np.full(self.atr_period-1, atr[0]), atr))
        logger.info(f"ATR calculated. First few values: {atr[:5]}")
        return atr
    
    def _open_position(self, price, direction):
        current_atr = self.atr[self.current_step]  # Use indexing instead of iloc
        stop_loss_distance = 2 * current_atr * direction  # Use 2 * ATR for stop loss
        
        position_size = calculate_position_size(self.balance, self.risk_per_trade, price, stop_loss_distance, self.leverage)
        
        self.position = position_size * direction
        self.entry_price = price
        
        fee = self.transaction_fee_per_lot * (abs(self.position) / 100000)
        self.balance -= fee

        self.stop_loss = price - stop_loss_distance
        self.take_profit = price + (2 * abs(stop_loss_distance) * direction)  # 2:1 reward-to-risk ratio

        # Calculate actual risk
        actual_risk = abs(self.position * stop_loss_distance)
        actual_risk_percentage = (actual_risk / self.balance) * 100

        logger.info(f"Opened position: Size={self.position}, Entry Price={self.entry_price}, "
                    f"Stop Loss={self.stop_loss}, Take Profit={self.take_profit}, Fee={fee}, "
                    f"Balance={self.balance}, Actual Risk %={actual_risk_percentage:.2f}%")

    def _close_position(self, price):
        if self.position != 0:
            profit = (price - self.entry_price) * self.position
            fee = self.transaction_fee_per_lot * (abs(self.position) / 100000)
            old_balance = self.balance
            self.balance += profit - fee
            logger.info(f"Closed position: Profit={profit}, Fee={fee}, Old Balance={old_balance}, New Balance={self.balance}")
            self.position = 0
            self.entry_price = 0
            self.stop_loss = None
            self.take_profit = None

    def _calculate_reward(self):
        current_price = self.df.iloc[self.current_step]['close']
        if self.position != 0:
            unrealized_profit = (current_price - self.entry_price) * self.position
            estimated_close_fee = self.transaction_fee_per_lot * (abs(self.position) / 100000)
            if self.initial_balance == 0:
                logger.warning("Initial balance is 0, returning 0 reward")
                return 0
            reward = (unrealized_profit - estimated_close_fee) / self.initial_balance
            logger.info(f"Calculated reward: {reward}")
            return reward
        return 0
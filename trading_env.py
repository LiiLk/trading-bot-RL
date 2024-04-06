import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.graph_objects as go
class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, data_file, initial_balance=1000, max_position=1, transaction_cost_pct=0.01, reward_scaling=1e-4):
        super(TradingEnv, self).__init__()
        self.data = pd.read_csv(data_file)
        # Preprocess data
        self.data = self.data.dropna() # Drop rows with missing values

        # # Convert datetime column to numeric representation
        # self.data['Date'] = pd.to_datetime(self.data['Date'])
        # self.data['Date'] = (self.data['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        # Convert data types for efficiency
        self.data = self.data.astype({'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32', 'Volume': 'float32'})

        self.window_size = 30 
        self.state_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.state_shape = (self.window_size, len(self.state_features))

        self.action_space = gym.spaces.Discrete(3) # 3 possible actions: 0 for hold, 1 for buy, 2 for sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.state_shape, dtype=np.float32)
        
        self.initial_balance = initial_balance
        self.balance = initial_balance # Initial balance in USD
        self.max_position = max_position # Maximum position size in BTC
        self.holdings = 0 # Initial holdings in BTC
        self.transaction_cost_pct = transaction_cost_pct # Transaction cost percentage
        self.reward_scaling = reward_scaling # Scaling factor for reward calculation

        # For rendering
        self.buy_signals = []
        self.sell_signals = []
        self.prices = self.data['Close'].tolist()  # Prices for rendering
        self.current_step = None
  
    def reset(self, seed=None):
        # Reset environment state
        self.balance = self.initial_balance
        self.holdings = 0
        self.current_step = np.random.randint(0, len(self.data) - self.window_size)
        self.buy_signals, self.sell_signals = [], []
        self.episode_start = self.current_step 

        # Seed the random number generator if a seed is provided
        if seed is not None:
            np.random.seed(seed)
        return self.get_state(self.current_step), {}

    def step(self, action):
        current_price = self.data.iloc[self.current_step + self.window_size - 1]['Close']
        self.current_step += 1

        if action == 1:  # Buy
            self.buy_signals.append(self.current_step)
            buy_amount = self.balance / current_price
            self.holdings += buy_amount
            self.balance -= buy_amount * current_price * (1 + self.transaction_cost_pct)

        elif action == 2:  # Sell
            self.sell_signals.append(self.current_step)
            sell_amount = self.holdings
            self.balance += sell_amount * current_price * (1 - self.transaction_cost_pct)
            self.holdings = 0

        next_state = self.get_state(self.current_step)
        reward = self.calculate_reward()  # Call the calculate_reward
        done = self.current_step >= len(self.data) - self.window_size
        
        return next_state, reward * self.reward_scaling, done, {}

    def get_state(self, step):
        window_frame = self.data.iloc[step:step + self.window_size][self.state_features].values
        normalized_frame = window_frame / np.max(window_frame, axis=0)
        return normalized_frame

    def render(self, mode='human'):
        if mode == 'human':
            buy_prices = [self.prices[i] for i in self.buy_signals]
            sell_prices = [self.prices[i] for i in self.sell_signals]
            dates = pd.to_datetime(self.data['Date']).iloc[self.current_step:self.current_step+self.window_size]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=self.prices[self.current_step:self.current_step+self.window_size], name='Price'))
            fig.add_trace(go.Scatter(x=[dates[i] for i in self.buy_signals], y=buy_prices, mode='markers', name='Buy', marker=dict(color='green', size=10, symbol='triangle-up')))
            fig.add_trace(go.Scatter(x=[dates[i] for i in self.sell_signals], y=sell_prices, mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')))
            fig.update_layout(title='Trading Chart', xaxis_title='Date', yaxis_title='Price')
            fig.show()
    
    def calculate_reward(self):
        current_portfolio_value = self.balance + self.holdings * self.data.loc[self.episode_start + self.state_shape[0], 'Close']
        previous_portfolio_value = self.balance + self.holdings * self.data.loc[self.episode_start + self.state_shape[0] - 1, 'Close']

        unrealized_pl = current_portfolio_value - previous_portfolio_value

        # Reward for holding profitable position
        if self.holdings > 0:
            current_price = self.data.loc[self.episode_start + self.state_shape[0], 'Close']
            previous_price = self.data.loc[self.episode_start + self.state_shape[0] - 1, 'Close']
            if current_price > previous_price:
                holding_reward = (current_price - previous_price) * self.holdings
            else:
                holding_reward = 0
        else:
            holding_reward = 0
        
        if self.holdings < 0:
            current_price = self.data.loc[self.episode_start + self.state_shape[0], 'Close']
            previous_price = self.data.loc[self.episode_start + self.state_shape[0] - 1, 'Close']
            if current_price < previous_price:
                holding_penality = (previous_price - current_price) * abs(self.holdings)
            else:
                holding_penality = 0
        else:
            holding_penality = 0

        # Combine unrealized P&L and holding reward/penalty
        reward = unrealized_pl + holding_reward - holding_penality

        # Discourage frequent trading
        reward -= 0.0001 * (1 if self.holdings != 0 else 0)
        return reward

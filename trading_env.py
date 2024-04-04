import gym
import numpy as np
import pandas as pd
import plotly.graph_objects as go
class TradingEnv(gym.Env):
    def __init__(self, data_file, initial_balance=1000, max_position=1, transaction_cost_pct=0.01, reward_scaling=1e-4):
        self.data = pd.read_csv(data_file)
        # Preprocess data
        self.data = self.data.dropna() # Drop rows with missing values

        # # Convert datetime column to numeric representation
        # self.data['Date'] = pd.to_datetime(self.data['Date'])
        # self.data['Date'] = (self.data['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        # Convert data types for efficiency
        self.data = self.data.astype({'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32', 'Volume': 'float32'})

        self.window_size = 30 
        self.state_feature = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.state_shape = (self.window_size, len(self.state_feature))

        self.action_space = gym.spaces.Discrete(3) # 3 possible actions: 0 for hold, 1 for buy, 2 for sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.state_shape, dtype=np.float32)
        
        self.initial_balance = initial_balance
        self.balance = initial_balance # Initial balance in USD
        self.max_position = max_position # Maximum position size in BTC
        self.holdings = 0 # Initial holdings in BTC
        self.transaction_cost_pct = transaction_cost_pct # Transaction cost percentage
        self.reward_scaling = reward_scaling # Scaling factor for reward calculation
  
    def reset(self):
        # Reset environment state
        self.episode_start = np.random.randint(0, len(self.data) - self.state_shape[0])
        self.balance = self.initial_balance
        self.holdings = 0
        self.next_state = self.get_state(self.episode_start)
        return self.next_state

    def step(self, action):
        # Execute action and return next observation, reward, done, info
        current_price = self.data.loc[self.episode_start + self.window_size - 1, 'Close']  

        if action == 1:  # Buy
            amount = self.balance * self.max_position // current_price
            self.balance -= amount * current_price * (1 + self.transaction_cost_pct)
            self.holdings += amount
        elif action == 2:  # Sell
            amount = self.holdings
            self.balance += amount * current_price * (1 - self.transaction_cost_pct)
            self.holdings = 0

        self.episode_start += 1
        next_state = self.get_state(self.episode_start)
        reward = self.calculate_reward()
        done = self.episode_start >= (len(self.data) - self.window_size)

        return next_state, reward * self.reward_scaling, done, {}

    def get_state(self, start):
        # Corrigez cette ligne pour utiliser `.loc` ou accéder d'abord par colonne puis par ligne
        end = start + self.window_size - 1
        state = self.data.loc[start:end, self.state_feature].values
        # Normalisation par le maximum de chaque colonne pour la fenêtre donnée
        state_normalized = state / state.max(axis=0)
        return state_normalized
    
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

    
    def render(self, mode='human'):
        if mode == 'human':
            # Show balance evolution
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=self.price_history.index, 
                                         open=self.data['Open'], 
                                         high=self.data['High'], 
                                         low=self.data['Low'], 
                                         close=self.data['Close']))
            fig.add_trace(go.Scatter(x=self.price_history.index[self.buy_signals], 
                                     y=self.price_history.index[self.buy_signals], 
                                     mode='markers', 
                                     marker=dict(symbol='triangle-up', color='green', size=10)))
            fig.add_trace(go.Scatter(x=self.price_history.index[self.sell_signals], 
                                     y=self.price_history.index[self.sell_signals], 
                                     mode='markers', 
                                     marker=dict(symbol='triangle-down', color='red', size=10)))
            fig.show()

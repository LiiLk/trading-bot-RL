# trading_agent.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading_env import BitcoinTradingEnv
from collections import deque
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Load historical Bitcoin price and volume data
df = pd.read_csv('BTCUSDT_historical_data.csv')  # Assuming you have a CSV file with price and volume data

# Normalize the data
price_mean = df['Close'].mean()
price_std = df['Close'].std()
volume_mean = df['Volume'].mean()
volume_std = df['Volume'].std()
df['Open'] = (df['Open'] - price_mean) / price_std
df['High'] = (df['High'] - price_mean) / price_std
df['Low'] = (df['Low'] - price_mean) / price_std
df['Close'] = (df['Close'] - price_mean) / price_std
df['Volume'] = (df['Volume'] - volume_mean) / volume_std

# Create the environment and agent
env = BitcoinTradingEnv(df)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = TradingAgent(state_size, action_size)

# Training parameters
num_episodes = 100
batch_size = 32

# Lists to store rewards and balances for each episode
rewards_list = []
balances_list = []

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if len(agent.memory) > batch_size:
            agent.train(batch_size)
    
    rewards_list.append(total_reward)
    balances_list.append(info['balance'])
    
    print(f"Episode: {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, Balance: {info['balance']:.2f}")

# Plot the rewards and balances
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards_list)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode')

plt.subplot(1, 2, 2)
plt.plot(balances_list)
plt.xlabel('Episode')
plt.ylabel('Balance')
plt.title('Balance per Episode')

plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
from B1_data import FinancialDataForTrading
from trading_env import TradingEnv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Redirect logs to a file
sys.stdout = open('trading_results.txt', 'w')

def simple_moving_average_strategy(env, observation, sma_period=20):
    price = observation[2]  # Assuming the current price is the third element in the observation
    sma = env.df['close'].rolling(window=sma_period).mean().iloc[env.current_step]
    if price > sma:
        return 1  # Buy
    elif price < sma:
        return 2  # Sell
    else:
        return 0  # Hold

# Load the data
data_handler = FinancialDataForTrading(symbol='EURUSD=X', csv_file_path='eurusd_data.csv')
data = data_handler.load_data()
print(f'Raw data shape: {data.shape}\n')

# Create the trading environments
env_random = TradingEnv(data, risk_per_trade=0.01, leverage=2)
env_sma = TradingEnv(data, risk_per_trade=0.01, leverage=2)
# Run simulations
num_episodes = 5
num_steps = 1000

strategies = {
    "Random": (env_random, lambda env, obs: env.action_space.sample()),
    "Simple MA": (env_sma, simple_moving_average_strategy)
}

for strategy_name, (env, strategy) in strategies.items():
    print(f"\nRunning {strategy_name} strategy:")
    for episode in range(num_episodes):
        observation = env.reset()
        total_reward = 0
        
        for step in range(num_steps):
            action = strategy(env, observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        print(f'Episode {episode + 1}:')
        print(f'  Total Reward: {total_reward:.6f}')
        print(f'  Cumulative Return: {info["cumulative_return"]:.6f}')
        print(f'  Final Portfolio Value: {info["portfolio_value"]:.2f}')
        print(f'  Trades Executed: {info["trades_executed"]}')
        print(f'  Number of steps: {step + 1}')
        print()

print('Simulation complete.')
# Close the file and restore stdout
sys.stdout.close()
sys.stdout = sys.__stdout__
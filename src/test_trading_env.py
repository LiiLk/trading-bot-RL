import pandas as pd
from src.B1_data import FinancialDataForTrading
from src.trading_env import TradingEnv

# Load the data
data_handler = FinancialDataForTrading(symbol='EURUSD=X', csv_file_path='eurusd_data.csv')
data = data_handler.load_data()
print(f'Raw data shape: {data.shape}\n')

# Create the trading environment
env = TradingEnv(data)

# Run a simulation with a random trading strategy
num_episodes = 5
num_steps = 1000

for episode in range(num_episodes):
    observation = env.reset()
    total_reward = 0
    
    for step in range(num_steps):
        # Randomly choose an action for demo purposes
        action = env.action_space.sample()  # Random action: Hold, Buy, Sell, Close
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    # Print the results of the episode
    final_portfolio_value = env.balance + (env.position * env.df.iloc[env.current_step]['close'])
    print(f'Episode {episode + 1}:')
    print(f'  Total Reward: {total_reward:.2f}')
    print(f'  Final Portfolio Value: {final_portfolio_value:.2f}')
    print(f'  Final Balance: {env.balance:.2f}')
    print(f'  Final Position: {env.position:.2f}')
    print(f'  Number of steps: {step + 1}')
    print()

print('Simulation complete.')
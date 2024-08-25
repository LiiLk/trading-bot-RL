import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from trading_env import TradingEnv
from collections import deque
import random
import csv
from datetime import datetime
from tqdm import tqdm

class TradingMetrics:
    def __init__(self, env: TradingEnv):
      self.env = env
      self.initial_balance = env.initial_balance
      self.balance_history = []
      self.returns = []

    def reset(self):
       self.balance_history = []
       self.returns = []

    def update(self, balance):
        self.balance_history.append(balance)
        if len(self.balance_history) > 1:
            self.returns.append((balance - self.balance_history[-2]) / self.balance_history[-2])

    def total_returns(self):
       return (self.balance_history[-1] - self.initial_balance) / self.initial_balance

    def sharpe_ratio(self, risk_free_rate=0.01):
       if len(self.returns) < 2:
          return 0
       return (np.mean(self.returns) - risk_free_rate) / np.std(self.returns)
    
    def max_drawdown(self):
       peak = self.balance_history[0]
       max_dd = 0
       for balance in self.balance_history:
            if balance > peak:
             peak = balance
            dd = (peak - balance) / peak
            if dd > max_dd:
               max_dd = dd
       return max_dd
    
    def win_rate(self):
       wins = sum(1 for r in self.returns if r > 0)
       return wins / len(self.returns) if self.returns else 0
    
    def profit_factor(self):
       gains = sum(r for r in self.returns if r > 0)
       losses = sum(abs(r) for r in self.returns if r < 0)
       return gains / losses if losses != 0 else float('inf')
    
    def get_metrics(self): 
       return {
          "Total Return": f"{self.total_returns():.2%}",
          "Sharpe Ratio": f"{self.sharpe_ratio():.2f}",
          "Max Drawdown": f"{self.max_drawdown():.2%}",
          "Win rate": f"{self.win_rate():.2%}",
          "Profit Factor": f"{self.profit_factor():.2f}"
       }

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005  # Changed from 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state)
            target_f[0][action] = target
            loss = nn.MSELoss()(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def backtest(env: TradingEnv, agent, episodes=10, batch_size=64):
    metrics = TradingMetrics(env)
    results = []

    # Create a unique filename for this backtest run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"backtest_results_{timestamp}.csv"

    with open(log_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Episode", "Steps", "Final Balance", "Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate", "Profit Factor"])

        for episode in tqdm(range(episodes), desc="Episodes"):
            state = env.reset()
            metrics.reset()
            done = False
            step = 0

            episode_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                metrics.update(env.balance)

                episode_reward += reward
                step += 1
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

                if step % 1000 == 0:
                    agent.update_target_model()
                    print(f"Episode {episode+1}/{episodes}, Step {step}, Balance: {env.balance:.2f}")

            episode_metrics = metrics.get_metrics()
            results.append({
                "Episode": episode + 1,
                "Steps": step,
                "Final Balance": env.balance,
                **episode_metrics
            })

            csv_writer.writerow([
                episode + 1,
                step,
                env.balance,
                episode_metrics["Total Return"],
                episode_metrics["Sharpe Ratio"],
                episode_metrics["Max Drawdown"],
                episode_metrics["Win rate"],
                episode_metrics["Profit Factor"]
            ])

            print(f"Episode {episode+1}/{episodes} completed:")
            print(f"  Steps: {step}")
            print(f"  Final Balance: {env.balance:.2f}")
            print(f"  Episode Reward: {episode_reward:.2f}")
            print(f"  Total Return: {episode_metrics['Total Return']}")
            print(f"  Sharpe Ratio: {episode_metrics['Sharpe Ratio']}")
            print(f"  Max Drawdown: {episode_metrics['Max Drawdown']}")
            print(f"  Win Rate: {episode_metrics['Win rate']}")
            print(f"  Profit Factor: {episode_metrics['Profit Factor']}")
            print("--------------------")

            # Save the model every 10 episodes
            if (episode + 1) % 10 == 0:
                agent.save(f"dqn_model_episode_{episode+1}.pth")

    print(f"Backtest results saved to {log_filename}")
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and reduce the data
    data = pd.read_csv("A:/AI/Apprentisage ai/Project for Learning AI/trading-bot-RL/eurusd_data.csv")
    subset_size = len(data) // 10
    data_subset = data.iloc[-subset_size:].reset_index(drop=True)
    
    # Save the subset of data in a temporary file
    temp_csv_path = "temp_subset_data.csv"
    data_subset.to_csv(temp_csv_path, index=False)
    
    # Create the environment with the temporary file
    env = TradingEnv(temp_csv_path)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Reduce the number of episodes and increase the batch size
    results = backtest(env, agent, episodes=100, batch_size=64)

    # Delete the temporary file after use
    import os
    os.remove(temp_csv_path)
   
    # Print summary of results
    print("\nBacktest Summary:")
    print(f"Total Episodes: {len(results)}")
    print(f"Final Balance: {results[-1]['Final Balance']:.2f}")
    print(f"Best Episode: {max(results, key=lambda x: x['Final Balance'])}")
    print(f"Worst Episode: {min(results, key=lambda x: x['Final Balance'])}")
import gymnasium as gym
from trading_env_old import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
import optuna
import torch
import pandas as pd

def create_env():
    data_file_path = 'BTCUSDT_historical_data.csv'
    data = pd.read_csv(data_file_path)
    env = TradingEnv(data_file=data_file_path)
    #print("TradingEnv created successfully with data:", env.data.head())

    return env


def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99)
    }

def optimize_agent(n_trials=10, n_timesteps=100000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = DummyVecEnv([lambda: create_env()])
    env = gym.wrappers.TransformObservation(env, lambda obs: torch.tensor(obs, dtype=torch.float32, device=device))
    
    def optimize_ppo_wrapper(trial):
        model_hyperparams = optimize_ppo(trial)

        progress_bar_callback = ProgressBarCallback()
        model = PPO('MlpPolicy', env, device=device, verbose=1, **model_hyperparams)
        model.learn(total_timesteps=n_timesteps, callback=[EvalCallback(env, best_model_save_path='./best_model', eval_freq=5000), progress_bar_callback])
        
        eval_env = create_env()
        obs = eval_env.reset()
        total_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward
            if done:
                break
        eval_env.reset()
        return total_reward

    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_ppo_wrapper, n_trials=n_trials)
    optimized_params = study.best_params

    # Create the final model with optimized hyperparameters
    model = PPO('MlpPolicy', env, verbose=1, **optimized_params)
    model.learn(total_timesteps=n_timesteps, callback=EvalCallback(env, best_model_save_path='./best_model', eval_freq=5000))
    return model

model = optimize_agent(n_trials=10, n_timesteps=100000)

obs = model.env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = model.env.step(action)
    model.env.render()

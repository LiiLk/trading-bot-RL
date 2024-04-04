from trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import optuna

def create_env():
    return TradingEnv(data_file='BTCUSDT_historical_data.csv')

def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99)
    }

def optimize_agent(n_trials=10, n_timesteps=100000):
    env = DummyVecEnv([lambda: create_env()])
    model = PPO('MlpPolicy', env, verbose=0)

    study = optuna.create_study(direction='maximize')
    try:
        study.optimize(optimize_ppo, n_trials=n_trials)
        optimized_params = study.best_params
    except ValueError:
        print("All trials failed. Using default hyperparameters.")
        optimized_params = {}
    model.learn(total_timesteps=n_timesteps, callback=EvalCallback(env, best_model_save_path='./best_model'), **optimized_params)
    return model

model = optimize_agent(n_trials=10, n_timesteps=100000)

obs = model.env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = model.env.step(action)
    model.env.render()

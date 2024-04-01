from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from b3_environnement_trader import TradingEnv  # Importez votre classe d'environnement personnalisée
import pandas as pd


class CustomCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.env.render()
        return True


df = pd.read_csv('historical_data.csv')
df = df[['open', 'high', 'low', 'close', 'volume']]
# Assumons que `TradingEnv` est votre classe d'environnement personnalisée
env = TradingEnv(df)  # 'df' est votre DataFrame contenant les données de marché
env = Monitor(env)
# Envelopper l'environnement pour l'utilisation avec Stable Baselines
env = DummyVecEnv([lambda: env])

# Créer et entraîner l'agent
model = PPO("MlpPolicy", env, verbose=2, learning_rate=0.0001, clip_range=0.1, n_steps=128)
model.learn(total_timesteps=10000, callback=CustomCallback(check_freq=1000))

# Évaluation de l'agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Récompense moyenne: {mean_reward}, Écart-type: {std_reward}")

# Sauvegarde du modèle
model.save("ppo_trading_bot")
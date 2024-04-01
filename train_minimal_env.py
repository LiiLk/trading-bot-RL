import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from test_minimal_env import MinimalEnv

if __name__ == '__main__':
    env = MinimalEnv()
    env = DummyVecEnv([lambda: env])  # Utiliser DummyVecEnv pour compatibilité

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

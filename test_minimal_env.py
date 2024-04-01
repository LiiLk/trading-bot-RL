import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MinimalEnv(gym.Env):
    """Un environnement minimal pour tester."""
    metadata = {'render.modes': ['console']}
    
    def __init__(self):
        super(MinimalEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # Actions: 0 ou 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Observation: un nombre entre 0 et 1
        self.state = None

    def step(self, action):
        self.state = np.random.rand(1)
        reward = 1.0 if action == self.state.round() else -1.0
        done = False
        info = {}
        return self.state, reward, done, info, {'state': self.state}

    def reset(self, **kwargs):
        self.state = np.random.rand(1)
        return self.state, {'state': self.state}

    def render(self, mode='console'):
        if mode == 'console':
            print(f'State: {self.state}')

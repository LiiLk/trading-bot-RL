import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    """Un environnement personnalisé pour le trading de Bitcoin."""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        
        # Assume df is a pandas DataFrame with OHLCV data
        self.df = df
        self.current_step = 0
        self.trades = []
        self.balance = 100  # Balance initiale en dollars
        self.position = 0  # Position initiale en Bitcoin
        self.transaction_cost = 0.001  # Coût de transaction (0.1%)
        
        # L'action peut être de 0 à 2, où 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # L'observation contient les données OHLCV pour les 5 derniers pas de temps
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(5, len(df.columns)), dtype=np.float32)
        
    def _next_observation(self):
        #print(self.df.iloc[self.current_step:self.current_step + 5].values)
        # Retourne les 5 dernières barres à partir du DataFrame
        return self.df.iloc[self.current_step:self.current_step + 5].values

    def step(self, action):
        # Appliquer l'action de trading (0 = hold, 1 = buy, 2 = sell)
        current_data = self.df.iloc[self.current_step]
        
        if action == 1:  # buy
            self.balance -= current_data['close'] * (1 + self.transaction_cost)
            self.position += 1
            self.trades.append(('buy', current_data['close']))
        elif action == 2:  # sell
            self.balance += current_data['close'] * (1 - self.transaction_cost)
            self.position -= 1
            self.trades.append(('sell', current_data['close']))
        
        # Calculer la récompense après avoir exécuté l'action
        reward = self.position * (current_data['close'] - self.trades[-1][1]) if self.trades else 0
        
        self.current_step += 1
        
        done = self.current_step >= len(self.df) - 5  # Arrêter si nous sommes à la fin du DataFrame
        
        obs = self._next_observation()
        
        return obs, reward, done, {}, {}
    
    def reset(self, **kwargs):
        # Réinitialiser l'environnement pour un nouvel épisode
        self.current_step = 0
        self.balance = 100
        self.position = 0
        self.trades = []
        return self._next_observation(), {}

    def render(self, mode='human', close=False):
        # Rendu de l'environnement si nécessaire (par exemple affichage des trades)
        profit = self.balance + self.position * self.df.iloc[self.current_step]['close'] - 100
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Position: {self.position}')
        print(f'Profit: {profit}')

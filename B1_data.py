import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class FinancialDataForTrading:
    def __init__(self, symbol='EURUSD=X', start_date='2010-01-01', end_date='None', csv_file_path='eurusd_data.csv'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.csv_file_path= csv_file_path
        self.data = None
        self.scaler = MinMaxScaler()
        
    def download_and_save_data(self):
        #Download EUR/USD dataset from Yahoo Finance
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=self.start_date, end=self.end_date)
        # Reset index to have the right column
        self.data = self.data.reset_index()

        # Rename the column with right format 
        self.data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        #Save data into CSV file
        self.data.to_csv(self.csv_file_path, index=False)

        print(f"Données sauvegardées dans {self.csv_file_path}")
        
        return self.data

    def load_data(self):
        if os.path.exists(self.csv_file_path):
            self.data = pd.read_csv(self.csv_file_path)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            print(f"Données chargées depuis {self.csv_file_path}")
        else: 
            print(f"Fichier {self.csv_file_path} non trouvé. Téléchargement des données...")
            self.download_and_save_data()
        return self.data

    def preprocess_data(self):
        if self.data is None:
            raise ValueError("Les données n'ont pas été chargées. Appelez load_data() d'abord.")
        # Returns calculation
        self.data['returns'] = self.data['close'].pct_change()
        # we measure the volatility in the range of 20 periods
        self.data['volatility'] = self.data['returns'].rolling(windows=20).std

        self.data['rsi'] = self.calculate_rsi(self.data['close'])

        # Normalize columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'returns', 'volatility', 'rsi']
        self.data[numeric_columns] = self.scaler.fit_transform(self.data[numeric_columns])

        # Deleting lines with NAN values
        self.data = self.data.dropna()

        return self.data
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1+rs))

    def get_training_data(self, sequence_length):
        if self.data is None:
            raise ValueError("Les données n'ont pas été prétraitées. Appelez preprocess_data() d'abord.")
        
        data = self.data[['open', 'high', 'low', 'close', 'volume', 'returns', 'volatility', 'rsi']].values

        X = []
        y = []

        for i in range(len(data) - sequence_length):
            X.append(data[i:(i+sequence_length)])
            y.append(data[i+sequence_length, 3]) #Close price will be our target

            return np.array(X), np.array(y)

# main 
if __name__ == "__main__":
    data_handler = FinancialDataForTrading(symbol='EURUSD=X', start_date='2020-01-01', csv_file_path='eurusd_data.csv')
    data = data_handler.load_data()
    preprocessed_data = data_handler.preprocess_data()
    X, y = data_handler.get_training_data(sequence_length=20)
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"First few rows of preprocessed data:\n{preprocessed_data.head()}")
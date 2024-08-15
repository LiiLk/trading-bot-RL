import datetime
import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from ta import add_momentum_ta, add_trend_ta, add_volume_ta, add_volatility_ta

class FinancialDataForTrading:
    def __init__(self, symbol='EURUSD=X', csv_file_path='eurusd_data.csv'):
        self.symbol = symbol
        self.csv_file_path = csv_file_path
        self.data = None
        self.scaler = MinMaxScaler()

        
    def download_and_save_data(self):
        # Calculate start date (730 days ago)
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=730)
        # Download EUR/USD dataset from Yahoo Finance
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=start_date, end=end_date, interval="1h")
        
        if self.data.empty:
            raise ValueError("No data downloaded. Please check your internet connection and the validity of the symbol.")
        # Reset index to have the right column
        self.data = self.data.reset_index()
        
        # Check the columns we actually have
        print(f"Columns in downloaded data: {self.data.columns}")
        
        # Rename and select only the columns we need
        columns_mapping = {
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        self.data = self.data.rename(columns=columns_mapping)
        self.data = self.data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Save data into CSV file
        self.data.to_csv(self.csv_file_path, index=False)

        print(f"Data saved to {self.csv_file_path}")
        
        return self.data


    def load_data(self):
        if os.path.exists(self.csv_file_path):
            self.data = pd.read_csv(self.csv_file_path)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], utc=True)
            print(f"Data loaded from {self.csv_file_path}")
        else: 
            print(f"File {self.csv_file_path} not found. Downloading data...")
            self.download_and_save_data()
        if self.data.empty:
            raise ValueError("The loaded data is empty. Please check the CSV file or try downloading the data again.")
        return self.data
    
    def add_technical_indicators(self):
        # Add momentum indicators
        self.data = add_momentum_ta(self.data, high="high", low="low", close="close", volume="volume", fillna=True)

        # Add volume indicators
        self.data = add_volume_ta(self.data, high="high", low="low", close="close", volume="volume", fillna=True)

        # Add volatility indicators
        self.data = add_volatility_ta(self.data, high="high", low="low", close="close", fillna=True)

        # Add trend indicators
        self.data = add_trend_ta(self.data, high="high", low="low", close="close", fillna=True)

        # Add custom indicators if needed
        self.data['SMA_20'] = self.data['close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['close'].rolling(window=50).mean()

        # Print the list of added indicators
        new_columns = [col for col in self.data.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print("Added technical indicators:", new_columns)

    def preprocess_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        try:
            self.add_technical_indicators()

            self.data = self.data.dropna(axis=1, how='all')

            self.data = self.data.ffill()

            # Normalize all numeric columns
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_columns] = self.scaler.fit_transform(self.data[numeric_columns])

            # Deleting lines with NAN values
            self.data = self.data.dropna()

            print(f"Preprocessed data shape: {self.data.shape}")
            print(f"Columns after preprocessing: {self.data.columns.tolist()}")

            return self.data
        except Exception as e:
            print(f"An error occurred during preprocessing: {str(e)}")
            raise

    def get_training_data(self, sequence_length):
        if self.data is None:
            raise ValueError("Les données n'ont pas été prétraitées. Appelez preprocess_data() d'abord.")
        
        features = self.data.drop(['timestamp'], axis=1).values
        X = []
        y = []

        for i in range(len(features) - sequence_length):
            X.append(features[i:(i+sequence_length)])
            y.append(features[i+sequence_length, self.data.columns.get_loc('close')])

        return np.array(X), np.array(y)

# main 
if __name__ == "__main__":
    try:
        data_handler = FinancialDataForTrading(symbol='EURUSD=X', csv_file_path='eurusd_data.csv')
        data = data_handler.load_data()
        print(f"Raw data shape: {data.shape}")
        preprocessed_data = data_handler.preprocess_data()
        print(f"Preprocessed data shape: {preprocessed_data.shape}")
        X, y = data_handler.get_training_data(sequence_length=24)
        
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
        print(f"First few rows of preprocessed data:\n{preprocessed_data.head()}")
        print(f"Features used: {preprocessed_data.columns.tolist()}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

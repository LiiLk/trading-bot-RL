# Forex and Indices Trading Bot

An AI-powered trading bot for forex pairs and indices using reinforcement learning.

## Project Overview

This project aims to develop a trading bot that can generate a monthly return of 10% to 20% with a 1% risk management strategy(i hope so lol). The bot currently focuses on the EUR/USD currency pair but is designed to be adaptable for other forex pairs and indices.

## Project Progress

[▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 15%

## Features

- Data retrieval and preprocessing for EUR/USD hourly data
- Technical indicator calculation and analysis
- Reinforcement learning model for trading decisions
- Risk management implementation adhering to FTMO rules
- Adaptable for various forex pairs and indices

## Supported Instruments

- EUR/USD (currently implemented)
- More forex pairs and indices to be added

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LiiLk/trading-bot-RL.git
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage

To run the data preprocessing and model training, you will need to: 
```bash
python B1_data.py
```
Modify the symbol in the main execution to use a different FX/indices pair :
```python
data_handler = FinancialDataForTrading(symbol='SYMBOL', csv_file_path='data.csv')
#e.g : data_handler = FinancialDataForTrading(symbol='GBPUSD', csv_file_path='data.csv')
```
Replace 'SYMBOL' with the desired forex pair or index (e.g., 'GBPUSD=X', '^GSPC' for S&P 500).

## Project Structure

    B1_data.py: Data retrieval and preprocessing
    trading_env.py: Trading environment for reinforcement learning
    trading_agent.py: Implementation of the trading agent
    
## Dependencies

    Python 3.x
    pandas
    numpy
    yfinance
    scikit-learn
    tensorflow (or your chosen RL library)

## Contributing 
Despite being a hobby project (who can maybe make money lol), contributions, issues, and feature requests are always welcomed.

# License
MIT

## Contact
Khalil ABBIOUI - khalil.abbioui@gmail.com



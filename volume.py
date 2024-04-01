import requests
from datetime import datetime

# Define the endpoint URL
url = "https://fapi.binance.com/futures/data/takerlongshortRatio"

# Define the parameters
params = {
    'symbol': 'BTCUSDT',
    'period': '5m',  # The period you want to fetch the data for
    'limit': 5       # The number of data points you want to fetch
}

# Make the GET request to the Binance Futures API
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print('Failed to retrieve data:', response.status_code)

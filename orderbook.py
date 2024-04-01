import requests

# Define the endpoint URL
url = "https://api.binance.com/api/v3/depth"

# Define the trading pair and the limit
params = {
    'symbol': 'BTCUSDT',
    'limit': 1000  # maximum limit
}

# Make the GET request to the Binance API
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print('Failed to retrieve data:', response.status_code)

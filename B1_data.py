from binance.client import Client
import pandas as pd

# Initialisation du client Binance
client = Client()

# Définition des paramètres pour les données historiques
symbol = 'BTCUSDT'
interval = '15m'  # '1m' pour 1 minute; vous pouvez utiliser '5m' pour 5 minutes, '1h' pour 1 heure, etc.
start_str = '2021-01-01'
end_str = '2023-01-01'

# Récupération des chandeliers historiques
candles = client.get_historical_klines(symbol, interval, start_str, end_str)

# Création du DataFrame
columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore']
df = pd.DataFrame(candles, columns=columns)

# Conversion des timestamps en dates lisibles et ajustement des types de données
df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Sauvegarde des données dans un fichier CSV
df.to_csv('BTCUSDT_historical_data.csv', index=False)

print("Les données historiques ont été sauvegardées dans BTCUSDT_historical_data.csv")

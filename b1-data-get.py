from datetime import datetime
import os
import time
import requests
import pandas as pd
import logging

# Configuration du logging
logging.basicConfig(filename='data_collection.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Configuration de l'API Binance
BINANCE_API_URL = "https://api.binance.com"
KLINE_ENDPOINT = "/api/v3/klines"
SYMBOL = "BTCUSDT"

# Fonction pour récupérer les données historiques de Binance
def get_historical_data(symbol, interval, limit=1000):
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    url = BINANCE_API_URL + KLINE_ENDPOINT
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        logging.error(f"Erreur lors de la récupération des données historiques: {err}")
        return None
    # Transformation des données en DataFrame
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(data, columns=cols)
    
    # Conversion des timestamps en dates lisibles
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms').astype('int64') // 10**9
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms').astype('int64') // 10**9
    
    # Conversion des colonnes de prix et de volume en numérique
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    
    return df

# Nouvelle fonction pour ajouter des données au CSV existant
def append_to_csv(data, filename):
    header = not os.path.exists(filename)
    if data is not None:
        # Assurez-vous que le fichier existe avec l'en-tête avant d'exécuter ce code
        data.to_csv(filename, mode='a', header=header, index=False)
# Fonction pour récupérer les données de l'Order Book
# def get_order_book(symbol, limit=1000):
#     url = BINANCE_API_URL + "/api/v3/depth"
#     params = {'symbol': symbol, 'limit': limit}
#     response = requests.get(url, params=params)
#     data = response.json()
#     return data

# # Fonction pour récupérer les transactions récentes
# def get_recent_trades(symbol, limit=5000):
#     url = BINANCE_API_URL + "/api/v3/trades"
#     params = {'symbol': symbol, 'limit': limit}
#     response = requests.get(url, params=params)
#     data = response.json()
#     return data


    
# Boucle infinie pour récupérer et sauvegarder les données toutes les heures
while True:
    # La date et l'heure actuelles
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Collecte des données à {current_time}...")
    print(f"Collecte des données à {current_time}...")
    
    try:
        # Collecte des données
        historical_data = get_historical_data(SYMBOL, '1h')
        # order_book_data = get_order_book(SYMBOL)
        # recent_trades_data = get_recent_trades(SYMBOL)

        # # Transformation des données en DataFrame
        # order_book_df = pd.DataFrame(order_book_data)
        # recent_trades_df = pd.DataFrame(recent_trades_data)

        # Ajoutez de nouvelles données aux fichiers CSV existants
        append_to_csv(historical_data, 'historical_data.csv')
        # append_to_csv(order_book_df, 'order_book_data.csv')
        # append_to_csv(recent_trades_df, 'recent_trades_data.csv')

        logging.info("Les données ont été sauvegardées.")
        print("Les données ont été sauvegardées.")
    except Exception as e:
        logging.error(f"Erreur lors de la collecte des données: {e}")
        print("Erreur lors de la collecte des données:", e)

    # Attendre une heure avant de collecter les données à nouveau
    time.sleep(3600)

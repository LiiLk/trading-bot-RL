import requests
from datetime import datetime, timedelta
def fetch_order_book(symbol):
    url = "https://api.binance.com/api/v3/depth"
    params = {'symbol': symbol, 'limit': 100000}
    response = requests.get(url, params=params)
    return response.json()

def calculate_order_book_score(order_book):
    buybook_total = sum(float(bid[1]) for bid in order_book['bids'] if float(bid[1]) >= 5)
    print(f"QTY Buy : {buybook_total}")
    sellbook_total = sum(float(ask[1]) for ask in order_book['asks'] if float(ask[1]) >= 5)
    print(f"QTY Sell : {sellbook_total}")

    if sellbook_total > buybook_total:
        return -1  # Indicating SHORT
    elif buybook_total > sellbook_total:
        return 1   # Indicating LONG
    else:
        return 0   # No clear indication


# Fonction permettant de convertir UTC datetime en milliseconds
def datetime_to_milliseconds(dt):
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000.0)

# Fetch les trades recents
def fetch_recent_trades(symbol, start_time):
    url = "https://api.binance.com/api/v3/aggTrades"
    params = {
        'symbol': symbol,
        'startTime': datetime_to_milliseconds(start_time),  # Start time in milliseconds
        'endTime': datetime_to_milliseconds(datetime.utcnow())  # Current time in milliseconds
    }
    response = requests.get(url, params=params)
    return response.json()

# Fonction pour calculer le score de volume
def calculate_volume_score(recent_trades):
    buy_volume = sum(float(trade['q']) for trade in recent_trades if not trade['m'])
    sell_volume = sum(float(trade['q']) for trade in recent_trades if trade['m'])
    print(f"Last Hour Buy Volume: {buy_volume}")
    print(f"Last Hour Sell Volume: {sell_volume}")

    # Determine the volume score
    if sell_volume > buy_volume:
        return -1  # Indicating SHORT
    elif buy_volume > sell_volume:
        return 1   # Indicating LONG
    else:
        return 0   # No clear indication

def fetch_liquidation_data(symbol, start_time):
    url = "https://fapi.binance.com/fapi/v1/allForceOrders"
    params = {
        'symbol': symbol,
        'startTime': datetime_to_milliseconds(start_time),  # Start time in milliseconds
        'endTime': datetime_to_milliseconds(datetime.utcnow()),  # Current time in milliseconds
        'limit': 1000  # Le maximum autorisé par l'API
    }
    response = requests.get(url, params=params)
    
    # Vérifiez si la requête a réussi
    if response.status_code == 200:
        data = response.json()
        long_liquidations = sum(float(order['price']) * float(order['origQty'])
                                for order in data if order['side'] == 'SELL' and float(order['origQty']) >= 1)
        short_liquidations = sum(float(order['price']) * float(order['origQty'])
                                 for order in data if order['side'] == 'BUY' and float(order['origQty']) >= 1)
        return long_liquidations, short_liquidations
    else:
        print(f"Failed to fetch liquidation data: {response.status_code}")
        return 0, 0



symbol = 'BTCUSDT'
one_hour_ago = datetime.utcnow() - timedelta(hours=1)

# Récupération et calcul du score pour l'Order Book
order_book = fetch_order_book(symbol)
order_book_score = calculate_order_book_score(order_book)
# Récupération et calcul du score pour le volume des dernières transactions
recent_trades = fetch_recent_trades(symbol, one_hour_ago)
volume_score = calculate_volume_score(recent_trades)

# Récupération des données de liquidation et calcul du score de liquidation
long_liquidations, short_liquidations = fetch_liquidation_data(symbol, one_hour_ago)
liquidation_score = 0  # Initialisez le score de liquidation
if long_liquidations > 1_000_000 or short_liquidations > 1_000_000:
    if long_liquidations > short_liquidations:
        liquidation_score = 1
    elif short_liquidations > long_liquidations:
        liquidation_score = -1

# Affichage des résultats
print(f"Order Book Score: {order_book_score}")
print(f"Volume Score: {volume_score}")
print(f"Liquidation Score: {liquidation_score}")

# Combinez les scores pour prendre une décision de trading
combined_score = order_book_score + volume_score + liquidation_score
print(f"Combined Score: {combined_score}")

# Décidez si le score indique un LONG ou SHORT
trade_decision = 'HOLD'  # Décision par défaut
if combined_score > 0:
    trade_decision = 'LONG'
elif combined_score < 0:
    trade_decision = 'SHORT'

print(f"Trade Decision: {trade_decision}")
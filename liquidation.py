import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    if 'data' in data:
        event_data = data['data']
        #print(event_data)
        if 'e' in event_data and event_data['e'] == 'forceOrder':
            force_order_data = event_data['o']
            # Filter for BTCUSDT symbol
            if force_order_data['s'] == 'BTCUSDT':
                print(f"Liquidation Order for BTCUSDT: "
                      f"Side: {force_order_data['S']}, "
                      f"Price: {force_order_data['ap']}, "
                      f"Quantity: {force_order_data['q']} BTC")
            elif force_order_data['s'] == 'ETHUSDT':
                print(f"Liquidation Order for ETHUSDT: "
                      f"Side: {force_order_data['S']}, "
                      f"Price: {force_order_data['ap']}, "
                      f"Quantity: {force_order_data['q']} ETH")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, *args):
    print("### WebSocket Closed ###")

def on_open(ws):
    print("WebSocket Connected. Listening for liquidation orders...")

websocket_url = "wss://fstream.binance.com/stream?streams=!forceOrder@arr"

ws = websocket.WebSocketApp(websocket_url,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)

ws.run_forever()

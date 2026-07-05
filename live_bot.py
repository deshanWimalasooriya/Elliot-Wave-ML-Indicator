import time
import numpy as np
import pandas as pd
from binance.client import Client
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import ta
import os

print("1. System is starting...")

# Connect to Binance Client directly (Public API)
client = Client()

# Load the AI Model
try:
    model = load_model('advanced_elliott_wave_model.h5')
    print("✅ AI Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error: Model not found. ({e})")
    exit()

def get_live_data_and_predict():
    try:
        print("\n🔄 Fetching live data from Binance...")
        
        # Fetch the last 1000 candles to avoid scaling issues
        # Using 1-Hour Timeframe (Can be changed to 4HOUR or 1DAY for larger moves)
        klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1000 hours ago UTC")
        
        df = pd.DataFrame(klines, columns=[
            'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades', 
            'Taker_Buy_Base_Volume', 'Taker_Buy_Quote_Volume', 'Ignore'
        ])
        
        # Convert data types to float
        df['Close'] = df['Close'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        
        # Calculate Technical Indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
        
        df.dropna(inplace=True)
        
        # Extract the 4 features used during training
        features = ['Close', 'Volume', 'RSI', 'ATR']
        data = df[features].values
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Extract only the last 60 timesteps of data required for the AI model
        last_60_days = scaled_data[-60:]
        
        # Reshape data to 3D for LSTM format (1, 60, 4)
        X_test = np.array([last_60_days])
        
        # Make the Prediction
        predicted_scaled_price = model.predict(X_test, verbose=0)
        
        # Inverse transform the scaled value back to the real price
        # Since 'Close' is at index 0, we apply the inverse transform accordingly
        temp_array = np.zeros((1, 4))
        temp_array[0, 0] = predicted_scaled_price[0][0]
        real_predicted_price = scaler.inverse_transform(temp_array)[0, 0]
        
        current_price = df['Close'].iloc[-1]
        
        # Calculate price difference and percentage change
        price_difference = real_predicted_price - current_price
        percentage_change = (price_difference / current_price) * 100

        print("-" * 50)
        print(f"💰 Current BTC Price    : ${current_price:,.2f}")
        print(f"🎯 AI Predicted Price   : ${real_predicted_price:,.2f}")
        print(f"📊 Expected Market Move : {percentage_change:.2f}%")

        # Decision Logic: Execute only if the move is > 5% or < -5%
        if percentage_change >= 5.0:
            print("🚀 DECISION: HIGH PROBABILITY UP TREND (>5%). [EXECUTE LONG]")
        elif percentage_change <= -5.0:
            print("💥 DECISION: HIGH PROBABILITY DOWN TREND (>5%). [EXECUTE SHORT]")
        else:
            print("⏳ DECISION: Market Noise Detected (Move is less than 5%). [WAIT]")
        print("-" * 50)
        
    except Exception as e:
        print(f"⚠️ Error fetching data: {e}")

# Infinite Loop for Live Monitoring
if __name__ == "__main__":
    print("🚀 Live Trading Bot Started. (Press Ctrl+C to exit)")
    while True:
        get_live_data_and_predict()
        
        # Fetches new data and updates prediction every 15 minutes
        print("⏳ Next update in 15 minutes...")
        time.sleep(900) # 900 seconds = 15 minutes
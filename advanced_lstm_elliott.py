import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta # Technical Analysis Library

print("1. Binance API හරහා දත්ත බාගත කිරීම ආරම්භ කෙරේ...")

# Binance Client ආරම්භ කිරීම (Public Data සඳහා API Keys අවශ්‍ය නොවේ)
client = Client()

# 2020 සිට අද දක්වා 1-Hour කාල රාමුවේ දත්ත ලබා ගැනීම
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 Jan, 2020")

# දත්ත Pandas DataFrame එකකට ඇතුළත් කිරීම
df = pd.DataFrame(klines, columns=[
    'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
    'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades', 
    'Taker_Buy_Base_Volume', 'Taker_Buy_Quote_Volume', 'Ignore'
])

# මිල අගයන් Float (දශම) බවට පරිවර්තනය කිරීම
df['Close'] = df['Close'].astype(float)
df['High'] = df['High'].astype(float)
df['Low'] = df['Low'].astype(float)
df['Volume'] = df['Volume'].astype(float)

print("2. Feature Engineering (තාක්ෂණික දර්ශක එකතු කිරීම)...")

# RSI (Relative Strength Index) ගණනය කිරීම
df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

# ATR (Average True Range) - වෙළඳපල විචල්‍යතාවය මැනීම සඳහා
df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()

# හිස් අගයන් (NaN values) ඉවත් කිරීම
df.dropna(inplace=True)

# ආකෘතියට ලබා දීමට අවශ්‍ය ප්‍රධාන Features තෝරාගැනීම
# දැන් අපි ආකෘතියට 'Close', 'Volume', 'RSI', සහ 'ATR' යන 4ම එකවර ලබා දෙමු
features = ['Close', 'Volume', 'RSI', 'ATR']
data = df[features].values

print("3. දත්ත ප්‍රමිතිකරණය (Data Normalization)...")

# දත්ත 0 සහ 1 අතර පරාසයකට ගෙන ඒම
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

print("4. කාල කවුළු සැකසීම (Sliding Windows)...")
window_size = 60 # අතීත පැය 60 (Features 4ක් සමගින්)
X_train, y_train = [], []

for i in range(window_size, len(scaled_data)):
    X_train.append(scaled_data[i - window_size:i]) # සියලුම Features අඩංගු අතීත දත්ත 60
    y_train.append(scaled_data[i, 0]) # පුරෝකථනය කළ යුත්තේ ඊළඟ 'Close' මිලයි (Index 0)

X_train, y_train = np.array(X_train), np.array(y_train)

print(f"පුහුණු දත්ත කට්ටලයේ හැඩය: {X_train.shape} (Samples, Timesteps, Features)")
print("5. Multivariate LSTM ආකෘතිය ගොඩනැගීම...")

model = Sequential()

# පළමු LSTM ස්ථරය (බහු-මාන දත්ත ආදානය සඳහා)
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# දෙවන LSTM ස්ථරය
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))

# සැඟවුණු සම්බන්ධක ස්ථර (Dense Layers)
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1)) # අවසාන මිල පුරෝකථනය (Output)

# ආකෘතිය සම්පාදනය කිරීම (Compile)
model.compile(optimizer='adam', loss='mean_squared_error')

print("6. ආකෘතිය පුහුණු කිරීම (Training) ආරම්භ කෙරේ...")
# පුහුණු කිරීම (Batch size 64 සහ Epochs 20 යටතේ)
model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.1)

# ආකෘතිය සුරැකීම
model.save('advanced_elliott_wave_model.h5')
print("✅ අතිවිශිෂ්ටයි! ආකෘතිය සාර්ථකව පුහුණු කර 'advanced_elliott_wave_model.h5' ලෙස සුරැකිනි.")
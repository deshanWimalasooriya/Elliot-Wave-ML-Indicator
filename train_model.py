import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

print("දත්ත සැකසීම ආරම්භ කෙරේ...")

# 1. දත්ත ලබා ගැනීම (Load Data)
# පෙර පියවරෙන් බාගත කළ CSV ගොනුව භාවිතා කරන්න
df = pd.read_csv('BTCUSDT_1H_Historical.csv')

# අපට අවශ්‍ය වන්නේ 'Close' (අවසන් මිල) තීරුව පමණි
data = df[['Close']].values

# 2. දත්ත ප්‍රමිතිකරණය (Data Normalization)
# ස්නායුක ජාල සඳහා දත්ත 0 ත් 1 ත් අතර අගයකට ගෙන ඒම අත්‍යවශ්‍ය වේ
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. කාල කවුළු සැකසීම (Create Sliding Windows)
window_size = 60 # අතීත පැය 60 ක දත්ත මත පදනම්ව මීළඟ පැය පුරෝකථනය කිරීම
X_train, y_train = [], []

for i in range(window_size, len(scaled_data)):
    X_train.append(scaled_data[i - window_size:i, 0])
    y_train.append(scaled_data[i, 0])

# ලැයිස්තු (Lists) Numpy Arrays බවට හැරවීම
X_train, y_train = np.array(X_train), np.array(y_train)

# LSTM ආකෘතියට අවශ්‍ය පරිදි දත්ත ත්‍රිමාණ (3D) හැඩයකට (Samples, Time Steps, Features) හැඩගැස්වීම
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(f"පුහුණු දත්ත කට්ටලයේ හැඩය: {X_train.shape}")
print("LSTM ආකෘතිය ගොඩනැගීම ආරම්භ කෙරේ...")

# 4. LSTM වාස්තු විද්‍යාව ගොඩනැගීම (Build LSTM Architecture)
model = Sequential()

# පළමු LSTM ස්ථරය (Temporal Feature Extractor)
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2)) # Overfitting වැළැක්වීම සඳහා

# දෙවන LSTM ස්ථරය
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))

# සම්බන්ධක ස්ථර (Dense Layers)
model.add(Dense(units=25))
model.add(Dense(units=1)) # අවසාන පුරෝකථනය (Output)

# 5. ආකෘතිය සම්පාදනය සහ පුහුණු කිරීම (Compile and Train)
# සාමාන්‍ය Mean Squared Error (MSE) හානි ශ්‍රිතය භාවිතා වේ
model.compile(optimizer='adam', loss='mean_squared_error')

print("ආකෘතිය පුහුණු කිරීම (Training) ආරම්භ කෙරේ. මෙයට සුළු වේලාවක් ගත විය හැක...")
# Batch size 64 ක් සහ Epochs 20 ක් යටතේ පුහුණු කිරීම
model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.1)

# 6. පුහුණු කළ ආකෘතිය සුරැකීම (Save Model)
model.save('elliott_wave_lstm_model.h5')
print("ආකෘතිය සාර්ථකව 'elliott_wave_lstm_model.h5' ලෙස සුරැකිනි!")
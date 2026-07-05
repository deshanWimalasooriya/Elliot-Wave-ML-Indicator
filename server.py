import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

app = FastAPI()

# 1. AI ආකෘතිය (Model) මතකයට ලබා ගැනීම
print("AI Model එක පූරණය වෙමින් පවතී...")
try:
    model = load_model('advanced_elliott_wave_model.h5')
    print("✅ Model එක සාර්ථකව පූරණය විය!")
except:
    print("⚠️ අවවාදයයි: Model එක සොයාගත නොහැක. කරුණාකර .h5 ගොනුව ඇති ස්ථානය පරීක්ෂා කරන්න.")

# 2. LSTM සඳහා අවශ්‍ය දත්ත 60 රඳවා තබා ගැනීමේ මතක ගබඩාව (Buffer)
# උපරිම දිග 60ක් වූ පෝලිමක් (Deque) ලෙස මෙය සකසා ඇත
data_buffer = deque(maxlen=60)

# 3. JSON Data Schema එක නිර්මාණය කිරීම (Pydantic භාවිතයෙන්)
class TradingViewWebhook(BaseModel):
    symbol: str
    close: float
    rsi: float
    volume: float

# 4. Webhook Endpoint එක
@app.post("/webhook")
async def receive_webhook(data: TradingViewWebhook):
    # TradingView මඟින් එවන දත්ත ලබාගැනීම
    print(f"\n🔔 නව සංඥාවක් ලැබුණා! | කාසිය: {data.symbol}")
    print(f"මිල: {data.close} | RSI: {data.rsi} | Volume: {data.volume}")
    
    # නව දත්තය Buffer එකට ඇතුළත් කිරීම (මෙහිදී ATR අගයද අවශ්‍ය නම් එකතු කළ හැක)
    # දැනට අපි Close, Volume, RSI පමණක් ආකෘතියට දෙන බව උපකල්පනය කරමු
    current_features = [data.close, data.volume, data.rsi]
    data_buffer.append(current_features)
    
    # Buffer එකේ දත්ත 60ක් සම්පූර්ණ වී ඇත්දැයි පරීක්ෂා කිරීම
    if len(data_buffer) == 60:
        print("දත්ත 60 සම්පූර්ණයි. AI ආකෘතිය මඟින් පුරෝකථනය (Prediction) ආරම්භ කෙරේ...")
        
        # Buffer එක Numpy Array එකක් බවට පත් කිරීම
        input_data = np.array(data_buffer)
        
        # දත්ත ප්‍රමිතිකරණය කිරීම (Normalization - 0 ත් 1 ත් අතරට ගෙන ඒම)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_input = scaler.fit_transform(input_data)
        
        # LSTM ආකෘතියට අවශ්‍ය ත්‍රිමාණ (3D) හැඩයට දත්ත වෙනස් කිරීම (1, 60, 3)
        lstm_input = np.reshape(scaled_input, (1, scaled_input.shape[0], scaled_input.shape[1]))
        
        # AI ආකෘතිය මඟින් අනාගත මිල අනුමාන කිරීම
        predicted_price_scaled = model.predict(lstm_input)
        
        # අනුමාන කළ 0-1 අතර අගය නැවතත් සැබෑ මිල බවට (Inverse Transform) පත් කිරීම
        # (මෙහිදී Close price එක අදාළ scaler එක හරහා inverse කළ යුතුය)
        predicted_real_price = predicted_price_scaled[0][0] # සැබෑ පරිවර්තනය මෙහි යෙදිය යුතුය
        
        print(f"📈 AI පුරෝකථනය කළ මීළඟ අගය (Normalized): {predicted_real_price}")
        
        # -------------------------------------------------------------
        # මෙතැනින් ඉදිරියට අදාළ Trade එක Binance API හරහා දැමිය හැක
        # -------------------------------------------------------------
        
        return {"status": "success", "prediction": float(predicted_real_price)}
    
    else:
        # තවමත් දත්ත 60 සම්පූර්ණ වී නොමැති නම්
        needed = 60 - len(data_buffer)
        print(f"⏳ තවත් දත්ත {needed} ක් TradingView වෙතින් ලැබෙන තුරු රැඳී සිටී...")
        return {"status": "pending", "message": f"Waiting for {needed} more data points"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
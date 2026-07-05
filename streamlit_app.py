import streamlit as st
import time
import numpy as np
import pandas as pd
from binance.client import Client
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------
# Page Configuration
# -----------------
st.set_page_config(page_title="AI Crypto Bot", layout="wide", page_icon="📈")

# -----------------
# Cache the Model to prevent reloading on every rerun
# -----------------
@st.cache_resource
def get_model():
    try:
        model = load_model('advanced_elliott_wave_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading AI model: {e}")
        return None

@st.cache_resource
def get_binance_client():
    return Client()

model = get_model()
client = get_binance_client()

# -----------------
# Sidebar UI & Controls
# -----------------
st.sidebar.title("🤖 Bot Settings")

coin_pair = st.sidebar.selectbox("Select Coin Pair", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"])

timeframe_map = {
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR
}
timeframe_label = st.sidebar.selectbox("Select Timeframe", ["15m", "1h", "4h"])
timeframe = timeframe_map[timeframe_label]

poll_interval = st.sidebar.slider("Update Interval (seconds)", min_value=10, max_value=300, value=60, step=10)

st.sidebar.markdown("---")

# Initialize session state variables
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0
if 'last_df' not in st.session_state:
    st.session_state.last_df = None
if 'last_current' not in st.session_state:
    st.session_state.last_current = None
if 'last_predicted' not in st.session_state:
    st.session_state.last_predicted = None

# Detect if user changed settings, force refresh
if 'last_coin' not in st.session_state or st.session_state.last_coin != coin_pair:
    st.session_state.last_coin = coin_pair
    st.session_state.last_update = 0
if 'last_timeframe' not in st.session_state or st.session_state.last_timeframe != timeframe:
    st.session_state.last_timeframe = timeframe
    st.session_state.last_update = 0

# Start / Stop Bot Logic
col1, col2 = st.sidebar.columns(2)
if col1.button("▶️ Start Bot", type="primary", use_container_width=True):
    st.session_state.bot_running = True
    st.session_state.last_update = 0 # Force immediate update on start
if col2.button("⏹️ Stop Bot", use_container_width=True):
    st.session_state.bot_running = False

st.sidebar.markdown(f"**Status:** {'🟢 RUNNING' if st.session_state.bot_running else '🔴 STOPPED'}")
if st.session_state.bot_running:
    st.sidebar.caption(f"Will refresh automatically every {poll_interval}s.")

# -----------------
# Core Functions
# -----------------
def fetch_data(symbol, interval):
    """Fetch recent historical klines from Binance."""
    # 200 candles ensures we have enough for 60-period LSTM sequence + 14-period indicators
    klines = client.get_klines(symbol=symbol, interval=interval, limit=200)
    df = pd.DataFrame(klines, columns=[
        'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades', 
        'Taker_Buy_Base_Volume', 'Taker_Buy_Quote_Volume', 'Ignore'
    ])
    
    # Convert types
    df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        
    return df

def process_data_and_predict(df):
    """Calculate indicators and run AI prediction."""
    # Calculate RSI & ATR
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    df_clean = df.dropna().copy()
    
    # Prepare features for the model
    features = ['Close', 'Volume', 'RSI', 'ATR']
    data = df_clean[features].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    if len(scaled_data) < 60:
        return None, None, None
        
    # Get last 60 steps for prediction
    last_60_days = scaled_data[-60:]
    X_test = np.array([last_60_days])
    
    # Predict
    predicted_scaled_price = model.predict(X_test, verbose=0)
    
    # Inverse transform to get real price
    temp_array = np.zeros((1, 4))
    temp_array[0, 0] = predicted_scaled_price[0][0]
    real_predicted_price = scaler.inverse_transform(temp_array)[0, 0]
    
    current_price = df_clean['Close'].iloc[-1]
    
    return current_price, real_predicted_price, df_clean

def create_plotly_chart(df):
    """Generate the candlestick chart with RSI and ATR subplots."""
    chart_df = df.tail(100) # Only display the latest 100 candles
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
    
    # Row 1: Candlesticks
    fig.add_trace(go.Candlestick(x=chart_df['Open_Time'],
                                 open=chart_df['Open'],
                                 high=chart_df['High'],
                                 low=chart_df['Low'],
                                 close=chart_df['Close'],
                                 name="Price"),
                  row=1, col=1)
                  
    # Row 2: RSI
    fig.add_trace(go.Scatter(x=chart_df['Open_Time'], y=chart_df['RSI'], 
                             line=dict(color='#bb86fc', width=2), name='RSI'),
                  row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#cf6679", row=2, col=1) # Overbought
    fig.add_hline(y=30, line_dash="dash", line_color="#03dac6", row=2, col=1) # Oversold
                  
    # Row 3: ATR
    fig.add_trace(go.Scatter(x=chart_df['Open_Time'], y=chart_df['ATR'], 
                             line=dict(color='#ffb74d', width=2), name='ATR'),
                  row=3, col=1)
                  
    fig.update_layout(title=f"<b>{coin_pair}</b> - Interactive Analysis",
                      xaxis_rangeslider_visible=False,
                      height=750,
                      template="plotly_dark",
                      margin=dict(l=40, r=40, t=60, b=40))
                      
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="ATR", row=3, col=1)
    
    return fig

# -----------------
# Main Dashboard
# -----------------
st.title(f"📈 AI Crypto Trading Bot - {coin_pair}")

# Placeholders for rendering content
metrics_placeholder = st.empty()
chart_placeholder = st.empty()

def update_data():
    """Fetch new data and run predictions."""
    if not model:
        return
        
    df = fetch_data(coin_pair, timeframe)
    current, predicted, df_clean = process_data_and_predict(df)
    
    if current is not None:
        st.session_state.last_df = df_clean
        st.session_state.last_current = current
        st.session_state.last_predicted = predicted
        st.session_state.last_update = time.time()

def render_dashboard(df, current, predicted):
    """Render the cached data to the Streamlit UI."""
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        
        # Format decimal places based on price magnitude (e.g., SHIB vs BTC)
        decimals = 2 if current > 1 else 6
        
        col1.metric("Current Price", f"${current:,.{decimals}f}")
        
        delta_price = predicted - current
        col2.metric("AI Predicted Price", f"${predicted:,.{decimals}f}", f"${delta_price:,.{decimals}f}")
        
        if predicted > current:
            col3.markdown("""
            <div style="background-color:rgba(3,218,198,0.1);padding:15px;border-radius:8px;border:1px solid #03dac6;text-align:center;">
                <h3 style="color:#03dac6;margin:0;">🚀 LONG (UP)</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            col3.markdown("""
            <div style="background-color:rgba(207,102,121,0.1);padding:15px;border-radius:8px;border:1px solid #cf6679;text-align:center;">
                <h3 style="color:#cf6679;margin:0;">📉 SHORT (DOWN)</h3>
            </div>
            """, unsafe_allow_html=True)
            
    fig = create_plotly_chart(df)
    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{st.session_state.last_update}")


# -----------------
# Execution Loop
# -----------------
if st.session_state.bot_running:
    # Check if we need to poll for new data
    time_since_update = time.time() - st.session_state.last_update
    if time_since_update > poll_interval:
        with st.spinner("Fetching latest data & AI prediction..."):
            update_data()
            
    # Render the dashboard with current state
    if st.session_state.last_df is not None:
        render_dashboard(st.session_state.last_df, st.session_state.last_current, st.session_state.last_predicted)
        
    # Sleep briefly, then rerun the app (creates a non-freezing loop)
    time.sleep(1)
    st.rerun()

else:
    # If bot is stopped, just render the last known state (if any)
    if st.session_state.last_df is not None:
        render_dashboard(st.session_state.last_df, st.session_state.last_current, st.session_state.last_predicted)
    else:
        st.info("Bot is idle. Click **'▶️ Start Bot'** in the sidebar to begin live AI predictions.")

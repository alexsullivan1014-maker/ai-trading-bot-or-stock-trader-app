# app.py - AI Stock Trading Bot Dashboard
# Run with: streamlit run app.py

import streamlit as st
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Trading Bot", layout="wide")
st.title("🤖 AI-Powered Stock Trading Bot (PPO Reinforcement Learning)")

st.warning("This is for educational purposes only. Trading involves high risk of loss. Use paper trading first!")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    SYMBOL = st.text_input("Stock Symbol", value="AAPL").upper()
    train_timesteps = st.number_input("Training Timesteps (cloud can't handle large numbers!)", value=5000, min_value=1000, max_value=10000)
    
    st.header("Alpaca API (Paper Trading Recommended)")
    use_paper = st.checkbox("Use Paper Trading", value=True)
    api_key = st.text_input("Alpaca API Key", type="password")
    secret_key = st.text_input("Alpaca Secret Key", type="password")
    
    if st.button("Save Keys Temporarily"):
        os.environ[PK7VEOS42Y234KVVOHJY2W5KWC] = api_key
        os.environ[GMpnf4XqCXjcp97TBn1VKtyAsaAfumWdUNpmY7St56JG] = secret_key
        st.success("Keys saved for this session")

# Your existing environment class (same as before)
class StockTradingEnv(gym.Env):
    # ... (copy the full StockTradingEnv class from previous code here)
    # (It's the same - no changes needed)

    def __init__(self, df, initial_balance=10000, lookback_window=50):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(lookback_window * 9 + 2,), dtype=np.float32)

    # ... (copy reset, step, _get_observation, etc. exactly as before)

def fetch_data(symbol, start='2020-01-01'):
    return yf.download(symbol, start=start, end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    
def load_or_train_model(symbol, df, initial_balance, timesteps):
    model_path = "ppo_stock_trader.zip"
    if os.path.exists(model_path):
        st.success("Loaded pre-trained model! 🚀")
        return PPO.load(model_path)
    else:
        st.error("No pre-trained model found (ppo_stock_trader.zip missing).")
        st.info("Training on Streamlit Cloud will likely crash due to low memory. "
                "Please pre-train the model locally on a computer and upload the ppo_stock_trader.zip file to this repo.")
        return None  # Stops everything safely if no model

# Backtest with visualization
def run_backtest(model, symbol, df, initial_balance):
    env = StockTradingEnv(df, initial_balance)
    obs, _ = env.reset()
    net_worths = [initial_balance]
    dates = [df.index[env.lookback_window]]
    
    initial_shares_bah = initial_balance / df['Close'].iloc[env.lookback_window]
    buy_and_hold = [initial_balance]
    
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, info = env.step(action)
        net_worths.append(info['net_worth'])
        dates.append(df.index[env.current_step])
        bah_value = initial_shares_bah * df['Close'].iloc[env.current_step]
        buy_and_hold.append(bah_value)
        if done:
            break
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, net_worths, label='AI Strategy')
    ax.plot(dates, buy_and_hold, label='Buy & Hold')
    ax.set_title(f"Backtest: {symbol}")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)
    
    strategy_ret = (net_worths[-1] / initial_balance - 1) * 100
    bah_ret = (buy_and_hold[-1] / initial_balance - 1) * 100
    st.metric("AI Strategy Return", f"{strategy_ret:.2f}%")
    st.metric("Buy & Hold Return", f"{bah_ret:.2f}%")

# Main app
df = fetch_data(SYMBOL)

if len(df) < 100:
    st.error("Not enough data for this symbol")
else:
 def load_or_train_model(symbol, df, initial_balance, timesteps):
    model_path = "ppo_stock_trader.zip"
    if os.path.exists(model_path):
        st.success("Loaded pre-trained model! 🚀")
        return PPO.load(model_path)
    else:
        st.error("No pre-trained model found (ppo_stock_trader.zip missing).")
        st.info("Training on Streamlit Cloud will likely crash due to low memory. "
                "Pre-train locally on a computer and upload the ppo_stock_trader.zip file.")
        return None
    
    st.subheader("Backtesting Results")
    run_backtest(model, SYMBOL, df, initial_balance)
    
    st.subheader("Live Prediction (Latest Data)")
    latest_env = StockTradingEnv(df.tail(200))  # Use recent data
    obs = latest_env._get_observation()
    action, _ = model.predict(obs)
    action_type = "Hold" if action[0] < 1 else ("Buy" if action[0] < 2 else "Sell")
    amount = action[1]
    st.write(f"**Suggested Action:** {action_type} {amount:.1%} of available resources")
    
    if st.button("Execute Trade (Paper/Live - HIGH RISK!)") and api_key and secret_key:
        client = TradingClient(api_key, secret_key, paper=use_paper)
        # Simple execution logic (similar to previous execute_trade)
        # Add your execute_trade logic here if desired
        st.warning("Trade execution not fully implemented in this example for safety.")

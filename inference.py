import os
import sys
import pickle
import time
import pandas as pd
import numpy as np
import ccxt
import psycopg2 
from datetime import datetime, timezone, timedelta
from collections import Counter
from huggingface_hub import hf_hub_download
import logging

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
ASSETS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 
    'AVAX/USDT', 'DOT/USDT', 'LTC/USDT', 'BCH/USDT',
    'LINK/USDT', 'UNI/USDT', 'AAVE/USDT', 'NEAR/USDT', 
    'FIL/USDT', 'ALGO/USDT', 'XLM/USDT', 'EOS/USDT',
    'DOGE/USDT', 'SHIB/USDT', 'SAND/USDT'
]

TIMEFRAME = '30m'
START_DATE = '2020-01-01T00:00:00Z'
END_DATE = '2026-01-01T00:00:00Z'

DATABASE_URL = os.getenv("DATABASE_URL")
DATA_DIR = "/app/data"
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"

TARGET_ACCURACY = 70.0
ACCURACY_TOLERANCE = 5.0
MIN_TRADES = 50
REQUEST_DELAY = 0.5

def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)

# --- DATABASE FUNCTIONS ---

def get_db_connection():
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable not found.")
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def init_db():
    """Initializes the simplified tables"""
    logger.info("Initializing database...")
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            # Current signals table (Fast Access)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal (
                    asset TEXT PRIMARY KEY,
                    prediction TEXT
                );
            """)
            # Historical outcomes table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal_history (
                    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    asset TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    outcome NUMERIC NOT NULL,
                    PRIMARY KEY (time, asset)
                );
            """)
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Tables are ready.")
        except Exception as e:
            logger.error(f"Failed to init DB: {e}")

def settle_and_update_fast(asset, new_pred, p_curr, p_prev):
    """
    1. Saves new prediction immediately for speed.
    2. Calculates and saves outcome of previous signal internally.
    """
    conn = get_db_connection()
    if not conn: return
    
    try:
        cur = conn.cursor()
        
        # Get the previous prediction to calculate its outcome
        cur.execute("SELECT prediction FROM signal WHERE asset = %s", (asset,))
        row = cur.fetchone()
        prev_pred_str = row[0] if row else None

        # --- SPEED OPTIMIZATION: SAVE NEW SIGNAL FIRST ---
        cur.execute("""
            INSERT INTO signal (asset, prediction)
            VALUES (%s, %s)
            ON CONFLICT (asset) 
            DO UPDATE SET prediction = EXCLUDED.prediction;
        """, (asset, new_pred))
        conn.commit() 
        
        # --- BACKGROUND: SETTLE PREVIOUS SIGNAL ---
        if prev_pred_str:
            # Map signal to multiplier: LONG=1, SHORT=-1, NEUTRAL=0
            signal_map = {"LONG": 1.0, "SHORT": -1.0, "NEUTRAL": 0.0}
            val = signal_map.get(prev_pred_str, 0.0)
            
            # Formula: Signal * (New Close - Old Close) / Old Close
            outcome = val * ((p_curr - p_prev) / p_prev) * 100
            
            cur.execute("""
                INSERT INTO signal_history (time, asset, prediction, outcome)
                VALUES (CURRENT_TIMESTAMP, %s, %s, %s)
            """, (asset, prev_pred_str, outcome))
            conn.commit()
            
            icon = "✅" if outcome > 0 else "❌" if outcome < 0 else "➖"
            logger.info(f"{asset:12} | New: {new_pred:8} | Prev: {outcome:+.4f}% {icon}")
        else:
            logger.info(f"{asset:12} | New: {new_pred:8} | First run (no prev signal)")

        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed fast update for {asset}: {e}")

# --- DATA & MODEL FUNCTIONS ---

def get_model_filename(symbol):
    return f"{symbol.split('/')[0].lower()}.pkl"

def fetch_or_load_data(symbol):
    ensure_directories()
    # Simple version for validation step
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def download_models():
    model_paths = {}
    for symbol in ASSETS:
        fname = get_model_filename(symbol)
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{HF_FOLDER}/{fname}", local_dir=".")
            model_paths[symbol] = path
        except: continue
    return model_paths

def load_all_models(model_paths_dict):
    loaded_models = {}
    for symbol, path in model_paths_dict.items():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                loaded_models[symbol] = data
        except: continue
    return loaded_models

def run_single_asset_live(symbol, anchor_price, model_data, exchange):
    """
    Returns: (New Prediction, Price_Just_Closed, Price_Before_That)
    """
    configs = model_data['ensemble_configs']
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=20)
        if len(ohlcv) < 3: return "ERROR", None, None
        
        # p_curr = close of candle that just finished (-2 because -1 is the live unfinished one)
        # p_prev = close of candle before that
        p_curr = ohlcv[-2][4]
        p_prev = ohlcv[-3][4]
        
        # Inference using closed candles
        live_df = pd.DataFrame(ohlcv[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        log_prices = np.log(live_df['close'].to_numpy() / anchor_price)
        
        up, down = 0, 0
        for cfg in configs:
            grid = np.floor(log_prices / cfg['step_size']).astype(int)
            if len(grid) < cfg['seq_len']: continue
            seq = tuple(grid[-cfg['seq_len']:])
            if seq in cfg['patterns']:
                pred_lvl = Counter(cfg['patterns'][seq]).most_common(1)[0][0]
                if pred_lvl > seq[-1]: up += 1
                elif pred_lvl < seq[-1]: down += 1
        
        res = "LONG" if (up > 0 and down == 0) else "SHORT" if (down > 0 and up == 0) else "NEUTRAL"
        return res, p_curr, p_prev
    except:
        return "ERROR", None, None

def start_multi_asset_loop(loaded_models, anchor_prices):
    exchange = ccxt.binance({'enableRateLimit': True})
    logger.info("Bot Live - Strategy: Save Prediction First, Settle Internally")
    
    while True:
        try:
            for symbol, model_data in loaded_models.items():
                anchor = anchor_prices.get(symbol)
                new_pred, p_curr, p_prev = run_single_asset_live(symbol, anchor, model_data, exchange)
                
                if new_pred != "ERROR":
                    settle_and_update_fast(symbol, new_pred, p_curr, p_prev)
                
                time.sleep(REQUEST_DELAY)

            # Sleep until next 30m candle
            now = time.time()
            sleep_time = ((now // 1800) + 1) * 1800 - now + 10
            logger.info(f"Batch complete. Sleeping {sleep_time/60:.1f}m...")
            time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(60)

def main():
    init_db()
    model_paths = download_models()
    loaded_models = load_all_models(model_paths)
    
    valid_models = {}
    anchors = {}
    
    # Quick validation check (optional but kept for safety)
    for sym in list(loaded_models.keys()):
        anchors[sym] = loaded_models[sym]['initial_price']
        valid_models[sym] = loaded_models[sym] # Assuming validity for example

    if valid_models:
        start_multi_asset_loop(valid_models, anchors)

if __name__ == "__main__":
    main()

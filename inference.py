import os
import sys
import pickle
import time
import pandas as pd
import numpy as np
import ccxt
import psycopg2 
import logging
import random
import json
from datetime import datetime, timezone, timedelta
from collections import Counter
from huggingface_hub import hf_hub_download

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
DATABASE_URL = os.getenv("DATABASE_URL")
DATA_DIR = "/app/data"
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"

REQUEST_DELAY = 0.5
ENTRY_TRACKER_FILE = "entry_prices.json"

def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)

# --- ENTRY PRICE TRACKING (NOT IN DB) ---

def load_entry_prices():
    if os.path.exists(ENTRY_TRACKER_FILE):
        try:
            with open(ENTRY_TRACKER_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_entry_prices(tracker):
    with open(ENTRY_TRACKER_FILE, 'w') as f:
        json.dump(tracker, f)

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
    """Initializes tables and renames 'outcome' to 'pnl'"""
    logger.info("Initializing database and migrating columns...")
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            # 1. Ensure tables exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal (
                    asset TEXT PRIMARY KEY,
                    prediction TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal_history (
                    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    asset TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    pnl NUMERIC NOT NULL,
                    PRIMARY KEY (time, asset)
                );
            """)
            # 2. Migration: Rename outcome to pnl if the old column exists
            cur.execute("""
                DO $$ 
                BEGIN 
                    IF EXISTS (SELECT 1 FROM information_schema.columns 
                               WHERE table_name='signal_history' AND column_name='outcome') THEN
                        ALTER TABLE signal_history RENAME COLUMN outcome TO pnl;
                    END IF;
                END $$;
            """)
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Database migration complete.")
        except Exception as e:
            logger.error(f"Failed to init DB: {e}")

def settle_and_update_fast(asset, new_pred, p_market, entry_tracker):
    """
    1. Updates signal state with timestamps.
    2. Calculates pnl from entry price (stored in json) and exit price.
    """
    conn = get_db_connection()
    if not conn: return
    
    try:
        cur = conn.cursor()
        
        # Check previous prediction from DB
        cur.execute("SELECT prediction FROM signal WHERE asset = %s", (asset,))
        row = cur.fetchone()
        prev_pred_str = row[0] if row else None
        
        # Calculate PNL if previous signal was valid and we have an entry price
        if prev_pred_str in ["LONG", "SHORT"] and asset in entry_tracker:
            p_entry = entry_tracker[asset]
            p_exit = p_market
            side = 1.0 if prev_pred_str == "LONG" else -1.0
            pnl = round(side * ((p_exit - p_entry) / p_entry) * 100, 6)
            
            cur.execute("""
                INSERT INTO signal_history (time, asset, prediction, pnl)
                VALUES (CURRENT_TIMESTAMP, %s, %s, %s)
            """, (asset, prev_pred_str, pnl))
            conn.commit() # Separate commit for history
            
            icon = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
            logger.info(f"{asset:12} | Exit: {p_exit:.4f} | PNL: {pnl:+.4f}% {icon}")

        # Update Current Signal Table
        start_t = datetime.now(timezone.utc)
        end_t = start_t + timedelta(minutes=30)
        cur.execute("""
            INSERT INTO signal (asset, prediction, start_time, end_time)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (asset) 
            DO UPDATE SET 
                prediction = EXCLUDED.prediction,
                start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time;
        """, (asset, new_pred, start_t, end_t))
        conn.commit() # Separate commit for signal update
        
        # Record NEW entry price locally
        entry_tracker[asset] = p_market
        save_entry_prices(entry_tracker)
        
        if not prev_pred_str:
            logger.info(f"{asset:12} | New: {new_pred:8} | Initial Run")

        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed update for {asset}: {e}")

# --- DATA & MODEL FUNCTIONS ---

def get_model_filename(symbol):
    return f"{symbol.split('/')[0].lower()}.pkl"

def download_models():
    model_paths = {}
    for symbol in ASSETS:
        fname = get_model_filename(symbol)
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{HF_FOLDER}/{fname}", local_dir=".")
            model_paths[symbol] = path
        except Exception as e:
            logger.warning(f"Could not download model for {symbol}: {e}")
            continue
    return model_paths

def load_all_models(model_paths_dict):
    loaded_models = {}
    for symbol, path in model_paths_dict.items():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                loaded_models[symbol] = data
        except Exception as e:
            logger.error(f"Failed to load model {symbol}: {e}")
            continue
    return loaded_models

def run_single_asset_live(symbol, anchor_price, model_data, exchange):
    configs = model_data['ensemble_configs']
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=20)
        if len(ohlcv) < 3: return "ERROR", None
        
        p_market = ohlcv[-2][4]  # Close of last completed candle
        
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
        return res, p_market
    except Exception as e:
        logger.debug(f"Inference error for {symbol}: {e}")
        return "ERROR", None

def start_multi_asset_loop(loaded_models, anchor_prices):
    exchange = ccxt.binance({'enableRateLimit': True})
    entry_tracker = load_entry_prices()
    logger.info("Bot Live - Frequency: 30m")
    
    while True:
        try:
            for symbol, model_data in loaded_models.items():
                anchor = anchor_prices.get(symbol)
                new_pred, p_market = run_single_asset_live(symbol, anchor, model_data, exchange)
                
                if new_pred != "ERROR":
                    settle_and_update_fast(symbol, new_pred, p_market, entry_tracker)
                
                time.sleep(REQUEST_DELAY)

            now = time.time()
            sleep_time = ((now // 1800) + 1) * 1800 - now + 10 
            logger.info(f"Cycle complete. Sleeping {sleep_time/60:.2f} minutes...")
            time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(60)

def main():
    ensure_directories()
    init_db()
    
    model_paths = download_models()
    loaded_models = load_all_models(model_paths)
    
    if not loaded_models:
        logger.error("No models were successfully loaded. Exiting.")
        return

    available_symbols = list(loaded_models.keys())
    sample_count = min(2, len(available_symbols))
    test_symbols = random.sample(available_symbols, sample_count)
    
    logger.info(f"--- STARTUP VALIDATION: {test_symbols} ---")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    validation_passed = True
    for sym in test_symbols:
        anchor = loaded_models[sym]['initial_price']
        pred, p_m = run_single_asset_live(sym, anchor, loaded_models[sym], exchange)
        if pred == "ERROR":
            validation_passed = False
            break
        logger.info(f"PRE-FLIGHT: {sym} OK. Price: {p_m}")

    if validation_passed:
        anchors = {sym: loaded_models[sym]['initial_price'] for sym in loaded_models}
        start_multi_asset_loop(loaded_models, anchors)
    else:
        logger.error("System halted due to validation failure.")

if __name__ == "__main__":
    main()

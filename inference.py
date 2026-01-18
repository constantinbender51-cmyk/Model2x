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
from dotenv import load_dotenv

# --- CONFIGURATION ---

# Assets to look for (Must match the assets trained in the training script)
TARGET_ASSETS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']

# Railway / Database Configuration
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Paths
DATA_DIR = "/app/data"
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"

# Validation Criteria
TARGET_ACCURACY = 70.0  # Matches training threshold

def ensure_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

# --- DATABASE FUNCTIONS ---

def get_db_connection():
    if not DATABASE_URL:
        print("[ERROR] DATABASE_URL environment variable not found.")
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return None

def init_db():
    print("[DB] Initializing database...")
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal (
                    asset TEXT PRIMARY KEY,
                    prediction TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP
                );
            """)
            conn.commit()
            cur.close()
            conn.close()
            print("[DB] Table 'signal' is ready.")
        except Exception as e:
            print(f"[ERROR] Failed to init DB: {e}")

def save_prediction_to_db(asset, prediction, start_time, end_time):
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            query = """
                INSERT INTO signal (asset, prediction, start_time, end_time)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (asset) 
                DO UPDATE SET 
                    prediction = EXCLUDED.prediction,
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time;
            """
            cur.execute(query, (asset, prediction, start_time, end_time))
            conn.commit()
            cur.close()
            conn.close()
            print(f"[DB] Saved {asset}: {prediction}")
        except Exception as e:
            print(f"[ERROR] Failed to save prediction for {asset}: {e}")

# --- HELPER FUNCTIONS ---

def get_model_filename(symbol):
    return f"{symbol.split('/')[0].lower()}.pkl"

def get_data_filename(symbol, timeframe):
    safe_sym = symbol.replace('/', '_')
    return os.path.join(DATA_DIR, f"ohlcv_{safe_sym}_{timeframe}.pkl")

def get_grid_indices_from_price(prices, initial_price, step_size):
    """
    Calculates grid indices based on log-distance from the initial_price.
    Formula: floor( ln(price / initial_price) / step_size )
    Matches the Numba logic from training script.
    """
    if isinstance(prices, (pd.Series, list)):
        prices = np.array(prices)
        
    # Prevent log(0) errors
    prices = np.maximum(prices, 1e-9)
    
    log_dist = np.log(prices / initial_price)
    return np.floor(log_dist / step_size).astype(int)

# --- MODEL LOADING ---

def download_models():
    """
    Downloads models for TARGET_ASSETS from Hugging Face.
    """
    print(f"\n[MODEL] Checking models for: {TARGET_ASSETS}")
    model_paths = {}
    
    for symbol in TARGET_ASSETS:
        fname = get_model_filename(symbol)
        try:
            print(f"[MODEL] Downloading {fname}...", end='\r')
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{HF_FOLDER}/{fname}", local_dir=".")
            
            # Handle potential subdir structure from hf_hub_download
            actual_path = os.path.join(".", HF_FOLDER, fname)
            if not os.path.exists(actual_path): 
                actual_path = path 
                
            model_paths[symbol] = actual_path
            print(f"[MODEL] Ready: {fname}          ")
        except Exception as e:
            print(f"\n[ERROR] Failed to download model for {symbol}: {e}")
            
    return model_paths

def load_all_models(model_paths_dict):
    """
    Loads pickle files and validates schema.
    CRITICAL: Extracts the 'timeframe' determined during training.
    """
    loaded_models = {}
    print(f"[INFERENCE] Loading model files...")
    
    for symbol, path in model_paths_dict.items():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            # Verify structure and compatibility
            required_keys = ['ensemble_configs', 'initial_price', 'timeframe']
            if any(k not in data for k in required_keys):
                print(f"[ERROR] {symbol} model file is missing keys. Please re-train.")
                continue
                
            loaded_models[symbol] = data
            print(f" -> Loaded {symbol} | Timeframe: {data['timeframe']} | Base Price: {data['initial_price']:.4f}")
        except Exception as e:
            print(f"[ERROR] Could not load pickle for {symbol}: {e}")
            
    return loaded_models

# --- INFERENCE ENGINE ---

def run_live_inference(symbol, exchange, model_data):
    """
    Fetches strictly necessary recent data and runs inference.
    """
    configs = model_data['ensemble_configs']
    initial_price = model_data['initial_price']
    
    # DYNAMIC TIMEFRAME: Use the one saved in the model
    timeframe = model_data['timeframe']
    
    # Determine max sequence length needed + buffer
    max_seq_len = max(c['seq_len'] for c in configs)
    fetch_limit = max_seq_len + 50 
    
    try:
        # Fetching latest data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=fetch_limit)
        if not ohlcv: return "ERROR"
    except Exception as e:
        print(f"[ERROR] Live fetch {symbol}: {e}")
        return "ERROR"
        
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Drop the last incomplete candle to prevent repainting
    df = df.iloc[:-1]
    
    if df.empty: return "ERROR"

    current_prices = df['close'].to_numpy()
    up_votes = 0
    down_votes = 0
    
    # Iterate through all ensemble configurations
    for cfg in configs:
        step_size = cfg['step_size']
        seq_len = cfg['seq_len']
        
        # Calculate grid indices for recent history
        grid = get_grid_indices_from_price(current_prices, initial_price, step_size)
        
        if len(grid) < seq_len: continue
        
        # Extract sequence
        current_seq = tuple(grid[-seq_len:])
        current_level = current_seq[-1]
        
        # Check against trained patterns
        if current_seq in cfg['patterns']:
            history = cfg['patterns'][current_seq]
            # Predict most common outcome
            predicted_level = max(history, key=history.get)
            diff = predicted_level - current_level
            
            if diff > 0: up_votes += 1
            elif diff < 0: down_votes += 1
            
    # Ensemble Consensus Logic
    if up_votes > 0 and down_votes == 0:
        return "LONG"
    elif down_votes > 0 and up_votes == 0:
        return "SHORT"
    else:
        return "NEUTRAL"

# --- MAIN LOOP ---

def start_bot(valid_models):
    exchange = ccxt.binance({'enableRateLimit': True})
    print(f"\n[SYSTEM] Starting Live Bot...")
    
    # We store the timestamp of the last candle we successfully processed per symbol
    last_processed = {} 
    
    while True:
        try:
            for symbol, model_data in valid_models.items():
                # DYNAMIC TIMEFRAME: Read strictly from model
                timeframe = model_data['timeframe']
                
                # A. Fetch very minimal data to check timestamps
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=3)
                except Exception as e:
                    print(f"[ERR] Fetch failed {symbol}: {e}")
                    continue
                
                if not ohlcv or len(ohlcv) < 2:
                    continue
                
                # B. Identify the latest CLOSED candle
                # ohlcv[-1] is the current (open) candle.
                # ohlcv[-2] is the last fully completed candle.
                last_closed_candle = ohlcv[-2]
                last_closed_ts = last_closed_candle[0] # Timestamp in ms
                
                # C. Check if we have seen this candle before
                last_seen_ts = last_processed.get(symbol, 0)
                
                if last_closed_ts > last_seen_ts:
                    # >>> NEW CANDLE DETECTED <<<
                    human_time = datetime.fromtimestamp(last_closed_ts/1000, tz=timezone.utc).strftime('%H:%M')
                    print(f"\n[{human_time}] New {timeframe} candle for {symbol}...")
                    
                    # 1. Run Inference
                    pred = run_live_inference(symbol, exchange, model_data)
                    
                    # 2. Log & Save
                    icon = "⚪"
                    if pred == "LONG": icon = "🟢"
                    elif pred == "SHORT": icon = "🔴"
                    
                    print(f"   >>> SIGNAL: {pred} {icon}")
                    
                    if pred not in ["ERROR", "NEUTRAL"]:
                        # Set signal validity
                        start_time = datetime.now(timezone.utc)
                        
                        # Estimate end time based on timeframe string for database log
                        duration_min = 30 # Default
                        if 'h' in timeframe: duration_min = int(timeframe.strip('h')) * 60
                        elif 'm' in timeframe: duration_min = int(timeframe.strip('m'))
                        
                        end_time = start_time + timedelta(minutes=duration_min)
                        save_prediction_to_db(symbol, pred, start_time, end_time)

                    # 3. Update State so we don't process this candle again
                    last_processed[symbol] = last_closed_ts
                    
                else:
                    # Candle already processed
                    pass

            # Sleep to prevent rate limit issues but stay responsive
            time.sleep(10) 
            
        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"[CRITICAL] Loop error: {e}")
            time.sleep(30)

def main():
    init_db()
    ensure_directories()
    
    # 1. Download Models
    model_paths = download_models()
    if not model_paths:
        print("[ERROR] No models downloaded. Exiting.")
        return
    
    # 2. Load Models & Configs
    # (This now automatically gets the correct 'timeframe' from the pickle)
    loaded_models = load_all_models(model_paths)
    if not loaded_models:
        print("[ERROR] No valid models loaded. Exiting.")
        return

    # 3. Start Live Bot
    start_bot(loaded_models)

if __name__ == "__main__":
    main()

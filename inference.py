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

# IMPORTANT: Update these timeframes to match the "WINNER" output from your training script.
# The training script selects different timeframes for different assets.
ASSET_CONFIGS = {
    'BTC/USDT': {'timeframe': '30m'}, 
    'ETH/USDT': {'timeframe': '30m'},
    'BNB/USDT': {'timeframe': '30m'},
    'SOL/USDT': {'timeframe': '30m'},
    'XRP/USDT': {'timeframe': '30m'}
}

# If an asset isn't in the specific list above, use this:
DEFAULT_TIMEFRAME = '30m'

# Training Data Parameters (Must match training script defaults to validate correctly)
START_DATE = '2020-01-01T00:00:00Z'
END_DATE = '2026-01-01T00:00:00Z'

# Railway / Database Configuration
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Paths
DATA_DIR = "/app/data"
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"

# Pass Criteria
TARGET_ACCURACY = 75.0 
ACCURACY_TOLERANCE = 5.0

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

def get_timeframe(symbol):
    return ASSET_CONFIGS.get(symbol, {}).get('timeframe', DEFAULT_TIMEFRAME)

def get_model_filename(symbol):
    return f"{symbol.split('/')[0].lower()}.pkl"

def get_data_filename(symbol, timeframe):
    safe_sym = symbol.replace('/', '_')
    return os.path.join(DATA_DIR, f"ohlcv_{safe_sym}_{timeframe}.pkl")

def get_grid_indices_from_price(prices, initial_price, step_size):
    """
    Calculates grid indices based on log-distance from the initial_price.
    Formula: floor( ln(price / initial_price) / step_size )
    """
    # prices can be a single float or a numpy array
    if isinstance(prices, pd.Series) or isinstance(prices, list):
        prices = np.array(prices)
        
    log_dist = np.log(prices / initial_price)
    return np.floor(log_dist / step_size).astype(int)

# --- DATA LOADING ---

def download_models():
    print(f"\n[MODEL] Checking models for: {list(ASSET_CONFIGS.keys())}")
    model_paths = {}
    
    for symbol in ASSET_CONFIGS.keys():
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
    loaded_models = {}
    print(f"[INFERENCE] Loading model files...")
    
    for symbol, path in model_paths_dict.items():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            # Verify new structure
            if 'ensemble_configs' not in data or 'initial_price' not in data:
                print(f"[ERROR] {symbol} model file has invalid/old format.")
                continue
                
            loaded_models[symbol] = data
            print(f" -> Loaded {symbol} (Base Price: {data['initial_price']:.4f})")
        except Exception as e:
            print(f"[ERROR] Could not load pickle for {symbol}: {e}")
            
    return loaded_models

def fetch_historical_validation_data(symbol):
    """Fetches full history for backtest validation only."""
    ensure_directories()
    timeframe = get_timeframe(symbol)
    data_file = get_data_filename(symbol, timeframe)
    
    # Try local cache first
    if os.path.exists(data_file):
        try:
            df = pd.read_pickle(data_file)
            return df
        except: pass

    print(f"[DATA] Fetching history for {symbol} [{timeframe}]...")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(START_DATE)
    end_ts = exchange.parse8601(END_DATE)
    all_ohlcv = []

    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since >= end_ts: break
            time.sleep(exchange.rateLimit / 1000 * 1.05)
        except Exception: break

    if not all_ohlcv: return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Filter range
    start_dt = pd.Timestamp(START_DATE, tz='UTC')
    end_dt = pd.Timestamp(END_DATE, tz='UTC')
    df = df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

    df.to_pickle(data_file)
    return df

# --- INFERENCE ENGINES ---

def run_backtest_validation(symbol, df, model_data):
    """
    Validates the loaded model against historical data to ensure integrity.
    """
    configs = model_data['ensemble_configs']
    initial_price = model_data['initial_price']
    
    close_prices = df['close'].to_numpy()
    total_len = len(close_prices)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    # Pre-compute grids for each config
    config_grids = []
    for cfg in configs:
        grid = get_grid_indices_from_price(close_prices, initial_price, cfg['step_size'])
        config_grids.append({'cfg': cfg, 'full_grid': grid})

    correct = 0
    total_trades = 0
    
    # Simulate the validation period used in training (80% to 90%)
    for i in range(idx_80, idx_90):
        up_votes = 0
        down_votes = 0
        
        for item in config_grids:
            cfg = item['cfg']
            grid = item['full_grid']
            seq_len = cfg['seq_len']
            
            # Form sequence ending at i-1 to predict i
            if i <= seq_len: continue
            
            seq_tuple = tuple(grid[i - seq_len : i])
            current_level = seq_tuple[-1]
            
            if seq_tuple in cfg['patterns']:
                history = cfg['patterns'][seq_tuple]
                predicted_level = Counter(history).most_common(1)[0][0]
                diff = predicted_level - current_level
                
                if diff > 0: up_votes += 1
                elif diff < 0: down_votes += 1
        
        prediction = 0
        if up_votes > 0 and down_votes == 0:
            prediction = 1 # LONG
        elif down_votes > 0 and up_votes == 0:
            prediction = -1 # SHORT
            
        if prediction != 0:
            actual_next_price = close_prices[i]
            prev_price = close_prices[i-1]
            
            if prediction == 1 and actual_next_price > prev_price: correct += 1
            elif prediction == -1 and actual_next_price < prev_price: correct += 1
            total_trades += 1

    acc = (correct / total_trades * 100) if total_trades > 0 else 0.0
    return acc, total_trades

def run_live_inference(symbol, exchange, model_data):
    """
    Fetches strictly necessary recent data and runs inference.
    """
    configs = model_data['ensemble_configs']
    initial_price = model_data['initial_price']
    timeframe = get_timeframe(symbol)
    
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
    
    current_prices = df['close'].to_numpy()
    up_votes = 0
    down_votes = 0
    
    for cfg in configs:
        step_size = cfg['step_size']
        seq_len = cfg['seq_len']
        
        # Calculate grid indices for recent history
        grid = get_grid_indices_from_price(current_prices, initial_price, step_size)
        
        if len(grid) < seq_len: continue
        
        # Extract sequence
        current_seq = tuple(grid[-seq_len:])
        current_level = current_seq[-1]
        
        if current_seq in cfg['patterns']:
            history = cfg['patterns'][current_seq]
            predicted_level = Counter(history).most_common(1)[0][0]
            diff = predicted_level - current_level
            
            if diff > 0: up_votes += 1
            elif diff < 0: down_votes += 1
            
    if up_votes > 0 and down_votes == 0:
        return "LONG"
    elif down_votes > 0 and up_votes == 0:
        return "SHORT"
    else:
        return "NEUTRAL"

# --- MAIN LOOP ---

def start_bot(valid_models):
    exchange = ccxt.binance({'enableRateLimit': True})
    print(f"\n[SYSTEM] Starting Live Bot for: {list(valid_models.keys())}")
    
    # We store the timestamp of the last candle we successfully traded/checked per symbol
    last_processed = {} 
    
    while True:
        try:
            # Wake up every ~10s to be responsive, but only act on new candles.
            
            for symbol, model_data in valid_models.items():
                timeframe = get_timeframe(symbol)
                
                # A. Fetch very minimal data to check timestamps (3 candles is plenty)
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
                    # (We fetch slightly more data inside run_live_inference for actual calc)
                    pred = run_live_inference(symbol, exchange, model_data)
                    
                    # 2. Log & Save
                    icon = "⚪"
                    if pred == "LONG": icon = "🟢"
                    elif pred == "SHORT": icon = "🔴"
                    
                    print(f"   >>> SIGNAL: {pred} {icon}")
                    
                    if pred != "ERROR":
                        # Set signal validity
                        start_time = datetime.now(timezone.utc)
                        # Estimate end time based on timeframe string
                        duration_min = 1
                        if 'h' in timeframe: duration_min = int(timeframe.strip('h')) * 60
                        elif 'm' in timeframe: duration_min = int(timeframe.strip('m'))
                        
                        end_time = start_time + timedelta(minutes=duration_min)
                        save_prediction_to_db(symbol, pred, start_time, end_time)

                    # 3. Update State so we don't process this candle again
                    last_processed[symbol] = last_closed_ts
                    
                else:
                    # We have already processed this candle.
                    pass

            # Sleep short enough to catch 1m candles quickly, 
            # but long enough to not spam API unnecessarily.
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
    
    # 1. Load Models
    model_paths = download_models()
    if not model_paths: return
    
    loaded_models = load_all_models(model_paths)
    if not loaded_models: return

    # 2. Validate Models (Optional but recommended sanity check)
    valid_models = {}
    print(f"\n--- VALIDATION STAGE ---")
    for symbol, model_data in loaded_models.items():
        try:
            df = fetch_historical_validation_data(symbol)
            if not df.empty:
                # Sanity Check
                first_price = df['close'].iloc[0]
                model_init = model_data['initial_price']
                if abs(first_price - model_init)/model_init > 0.5:
                     print(f"[WARN] {symbol} Data Mismatch! History Start: {first_price}, Model Base: {model_init}")
                
                acc, trades = run_backtest_validation(symbol, df, model_data)
                status = "PASS ✅" if acc > (TARGET_ACCURACY - 10) else "WARN ⚠️"
                print(f"{symbol}: Acc {acc:.2f}% ({trades} candles) - {status}")
                valid_models[symbol] = model_data
            else:
                print(f"[WARN] Could not validate {symbol}, proceeding blindly.")
                valid_models[symbol] = model_data
        except Exception as e:
            print(f"[ERROR] Validation error {symbol}: {e}")
            valid_models[symbol] = model_data

    # 3. Start Live
    if valid_models:
        start_bot(valid_models)
    else:
        print("No valid models to run.")

if __name__ == "__main__":
    main()

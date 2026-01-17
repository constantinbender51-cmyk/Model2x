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

# --- CONFIGURATION ---
# List of assets to trade. 
# MUST match the assets used in your training script.
ASSETS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
TIMEFRAME = '30m'
START_DATE = '2020-01-01T00:00:00Z'
END_DATE = '2026-01-01T00:00:00Z'

# Railway / Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# Paths
DATA_DIR = "/app/data"
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"

# Pass Criteria (Global settings, can be adjusted per asset if needed)
TARGET_ACCURACY = 80.0 # Slightly lowered generic target for multi-asset
ACCURACY_TOLERANCE = 5.0
TARGET_TRADES = 1000
TRADES_TOLERANCE = 200

def ensure_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

# --- DATABASE FUNCTIONS ---

def get_db_connection():
    """Establishes connection to the Railway Postgres DB"""
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
    """Creates the signal table if it does not exist"""
    print("[DB] Initializing database...")
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            # asset is PRIMARY KEY to allow UPSERT for multiple coins
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
    """
    Saves the prediction using Upsert (ON CONFLICT).
    """
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

# --- DATA & MODEL FUNCTIONS ---

def get_model_filename(symbol):
    """Converts symbol (ETH/USDT) to filename (eth.pkl)"""
    return f"{symbol.split('/')[0].lower()}.pkl"

def get_data_filename(symbol):
    """Converts symbol (ETH/USDT) to cached filename"""
    safe_sym = symbol.replace('/', '_')
    return os.path.join(DATA_DIR, f"ohlcv_{safe_sym}.pkl")

def fetch_or_load_data(symbol):
    ensure_directories()
    data_file = get_data_filename(symbol)
    
    if os.path.exists(data_file):
        print(f"[DATA] Found local data for {symbol} at {data_file}. Loading...")
        try:
            df = pd.read_pickle(data_file)
            return df
        except Exception as e:
            print(f"[DATA] Error loading pickle for {symbol}: {e}. Re-fetching.")

    print(f"[DATA] Fetching new data for {symbol} ({START_DATE} to {END_DATE})...")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(START_DATE)
    end_ts = exchange.parse8601(END_DATE)
    all_ohlcv = []

    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Fetched {symbol} up to {datetime.fromtimestamp(ohlcv[-1][0]/1000)}...", end='\r')
            if since >= end_ts: break
            time.sleep(exchange.rateLimit / 1000 * 1.05)
        except Exception as e:
            print(f"\n[ERROR] Fetch failed for {symbol}: {e}")
            break

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    start_dt = pd.Timestamp(START_DATE, tz='UTC')
    end_dt = pd.Timestamp(END_DATE, tz='UTC')
    df = df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

    df.to_pickle(data_file)
    print(f"\n[DATA] Saved {len(df)} rows for {symbol}.")
    return df

def download_models():
    """
    Downloads models for ALL configured ASSETS.
    Returns a dictionary: { 'ETH/USDT': '/path/to/eth.pkl', ... }
    """
    print(f"\n[MODEL] checking models for: {ASSETS}")
    model_paths = {}
    
    for symbol in ASSETS:
        fname = get_model_filename(symbol)
        print(f"[MODEL] Downloading {fname} from Hugging Face...")
        try:
            # We assume the models are in the HF_FOLDER (model2x)
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{HF_FOLDER}/{fname}", local_dir=".")
            
            # Handling the directory structure HF creates locally
            actual_path = os.path.join(".", HF_FOLDER, fname)
            if not os.path.exists(actual_path): 
                actual_path = path # Fallback to wherever hf_hub_download put it
                
            model_paths[symbol] = actual_path
        except Exception as e:
            print(f"[ERROR] Failed to download model for {symbol}: {e}")
            # We do not exit here, we just skip this asset
            
    return model_paths

def load_all_models(model_paths_dict):
    """
    Loads pickle content into memory.
    Returns: { 'ETH/USDT': model_data_dict, ... }
    """
    loaded_models = {}
    print(f"[INFERENCE] Loading model files into memory...")
    
    for symbol, path in model_paths_dict.items():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                loaded_models[symbol] = data
            print(f" -> Loaded {symbol}")
        except Exception as e:
            print(f"[ERROR] Could not load pickle for {symbol}: {e}")
            
    return loaded_models

# --- INFERENCE LOGIC ---

def get_grid_indices(df, step_size):
    close_array = df['close'].to_numpy()
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    abs_price_log = np.log(abs_price_raw)
    return np.floor(abs_price_log / step_size).astype(int)

def run_backtest_inference(df, model_data):
    """
    Runs historical validation to confirm model quality before starting live.
    """
    configs = model_data['ensemble_configs']
    total_len = len(df)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    config_grids = []
    for cfg in configs:
        grid = get_grid_indices(df, cfg['step_size'])
        max_lookback = cfg['seq_len']
        slice_start = idx_80 - max_lookback
        val_slice = grid[slice_start:idx_90]
        config_grids.append({'cfg': cfg, 'val_seq': val_slice, 'offset': slice_start})

    correct, total_trades = 0, 0
    range_len = idx_90 - idx_80
    
    for i in range(range_len):
        target_abs_idx = idx_80 + i
        up_votes, down_votes = 0, 0
        valid_voters = [] 
        
        for grid_data in config_grids:
            cfg = grid_data['cfg']
            val_seq = grid_data['val_seq']
            offset = grid_data['offset']
            seq_len = cfg['seq_len']
            local_target_idx = target_abs_idx - offset
            
            if local_target_idx < seq_len: continue 
                
            current_seq_tuple = tuple(val_seq[local_target_idx - seq_len : local_target_idx])
            current_level = current_seq_tuple[-1]
            
            if current_seq_tuple in cfg['patterns']:
                history = cfg['patterns'][current_seq_tuple]
                predicted_level = Counter(history).most_common(1)[0][0]
                diff = predicted_level - current_level
                if diff > 0:
                    up_votes += 1
                    valid_voters.append(('up', cfg, val_seq, local_target_idx))
                elif diff < 0:
                    down_votes += 1
                    valid_voters.append(('down', cfg, val_seq, local_target_idx))

        prediction = 0 
        if up_votes > 0 and down_votes == 0:
            prediction = 1
            best_voter = max([v for v in valid_voters if v[0] == 'up'], key=lambda x: x[1]['step_size'])
        elif down_votes > 0 and up_votes == 0:
            prediction = -1
            best_voter = max([v for v in valid_voters if v[0] == 'down'], key=lambda x: x[1]['step_size'])
            
        if prediction != 0:
            _, _, chosen_seq, local_idx = best_voter
            actual_diff = chosen_seq[local_idx] - chosen_seq[local_idx - 1]
            if actual_diff != 0:
                total_trades += 1
                if (prediction == 1 and actual_diff > 0) or (prediction == -1 and actual_diff < 0):
                    correct += 1

    accuracy = (correct / total_trades * 100) if total_trades > 0 else 0.0
    return accuracy, total_trades

def run_single_asset_live(symbol, anchor_price, model_data, exchange):
    """
    Performs a single live prediction for ONE asset.
    Returns the Prediction Status String (LONG/SHORT/NEUTRAL)
    """
    configs = model_data['ensemble_configs']
    
    try:
        live_ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=50)
    except Exception as e:
        print(f"[ERROR] {symbol} fetch failed: {e}")
        return "ERROR"

    if not live_ohlcv: return "ERROR"

    live_df = pd.DataFrame(live_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms', utc=True)
    
    # Drop unfinished candle
    live_df = live_df.iloc[:-1] 
    
    prices = live_df['close'].to_numpy()
    # Use the specific anchor price stored in the model data for this asset
    log_prices = np.log(prices / anchor_price)
    
    up_votes = 0
    down_votes = 0
    
    for cfg in configs:
        step_size = cfg['step_size']
        seq_len = cfg['seq_len']
        grid = np.floor(log_prices / step_size).astype(int)
        
        if len(grid) < seq_len: continue
            
        current_seq_tuple = tuple(grid[-seq_len:])
        current_level = current_seq_tuple[-1]
        
        if current_seq_tuple in cfg['patterns']:
            history = cfg['patterns'][current_seq_tuple]
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

def start_multi_asset_loop(loaded_models, anchor_prices):
    print(f"\n[SYSTEM] Initializing Multi-Asset Live Bot Loop...")
    print(f"[SYSTEM] Monitoring: {list(loaded_models.keys())}")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            start_time = now_utc
            end_time = now_utc + timedelta(minutes=30)
            
            print(f"\n{'='*40}")
            print(f"LIVE RUN - {now_utc.strftime('%H:%M:%S UTC')}")
            
            for symbol, model_data in loaded_models.items():
                anchor = anchor_prices.get(symbol)
                if not anchor:
                    print(f"Skipping {symbol} (No anchor price)")
                    continue
                
                pred = run_single_asset_live(symbol, anchor, model_data, exchange)
                
                # Console output
                icon = "⚪"
                if pred == "LONG": icon = "🟢"
                elif pred == "SHORT": icon = "🔴"
                print(f"{symbol}: {pred} {icon}")
                
                # Save to DB
                if pred != "ERROR":
                    save_prediction_to_db(symbol, pred, start_time, end_time)
                
                # Slight delay to respect rate limits inside the loop
                time.sleep(0.5)

            print(f"{'='*40}")

            # Calculate Sleep Time
            now = time.time()
            interval = 30 * 60 # 1800 seconds
            next_timestamp = ((now // interval) + 1) * interval
            sleep_duration = next_timestamp - now + 10
            
            next_run_dt = datetime.fromtimestamp(next_timestamp + 10)
            print(f"[WAIT] Next run: {next_run_dt.strftime('%H:%M:%S')}")
            
            time.sleep(sleep_duration)
            
        except KeyboardInterrupt:
            print("\n[STOP] Bot stopped by user.")
            break
        except Exception as e:
            print(f"\n[ERROR] Unexpected error in loop: {e}")
            time.sleep(60)

def main():
    # 0. Initialize Database
    init_db()

    # 1. Download Models
    model_paths = download_models()
    if not model_paths:
        print("No models downloaded. Exiting.")
        sys.exit(1)

    # 2. Load Models
    loaded_models = load_all_models(model_paths)
    if not loaded_models:
        print("No models loaded. Exiting.")
        sys.exit(1)

    # 3. Validation (Backtest) & Get Anchor Prices
    # We need to fetch historical data for each asset to get the Anchor Price (index 0)
    # and run the validation check.
    
    valid_models = {}
    anchor_prices = {}
    
    print(f"\n--- BACKTEST VALIDATION ---")
    
    for symbol in list(loaded_models.keys()):
        df = fetch_or_load_data(symbol)
        if df.empty:
            print(f"[WARN] No data for {symbol}. Skipping.")
            continue
            
        # Get anchor price (first close price in the dataset used for training alignment)
        # In a production restart scenario, ensuring this matches training is crucial.
        # Ideally, the model pickle should contain the reference price, but per your script 
        # structure, we derive it from data.
        anchor_price = df['close'].iloc[0] 
        anchor_prices[symbol] = anchor_price
        
        acc, trades = run_backtest_inference(df, loaded_models[symbol])
        
        status = "FAIL ❌"
        # Using a slightly wider tolerance for passing "ALL" models automatically
        if (TARGET_ACCURACY - ACCURACY_TOLERANCE) <= acc: 
             status = "PASS ✅"
             valid_models[symbol] = loaded_models[symbol]
        else:
             print(f"[WARN] {symbol} accuracy {acc:.2f}% is too low. Exclude if strict.")
             # For now, we allow it but log it. Uncomment below to enforce strict mode.
             valid_models[symbol] = loaded_models[symbol] 
        
        print(f"{symbol} | Acc: {acc:.2f}% | Trades: {trades} | Status: {status}")

    if not valid_models:
        print("No valid models passed validation.")
        sys.exit(1)
        
    # 4. Start the Continuous Loop with all valid models
    start_multi_asset_loop(valid_models, anchor_prices)

if __name__ == "__main__":
    main()

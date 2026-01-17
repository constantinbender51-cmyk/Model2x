import os
import sys
import pickle
import time
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
from collections import Counter
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---pee
SYMBOL = 'ETH/USDT'
TIMEFRAME = '30m'
START_DATE = '2020-01-01T00:00:00Z'
END_DATE = '2026-01-01T00:00:00Z'

# Paths
DATA_DIR = "/app/data"
DATA_FILE = os.path.join(DATA_DIR, "ohlcv_data.pkl")
MODEL_FILENAME = "eth.pkl"
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"

# Pass Criteria
TARGET_ACCURACY = 85.0
ACCURACY_TOLERANCE = 5.0
TARGET_TRADES = 1100
TRADES_TOLERANCE = 100

def ensure_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def fetch_or_load_data():
    ensure_directories()
    if os.path.exists(DATA_FILE):
        print(f"[DATA] Found local data at {DATA_FILE}. Loading...")
        try:
            df = pd.read_pickle(DATA_FILE)
            return df
        except Exception as e:
            print(f"[DATA] Error loading pickle: {e}. Re-fetching.")

    print(f"[DATA] Fetching new data from Binance ({START_DATE} to {END_DATE})...")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(START_DATE)
    end_ts = exchange.parse8601(END_DATE)
    all_ohlcv = []

    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Fetched up to {datetime.fromtimestamp(ohlcv[-1][0]/1000)}...", end='\r')
            if since >= end_ts: break
            time.sleep(exchange.rateLimit / 1000 * 1.05)
        except Exception as e:
            print(f"\n[ERROR] Fetch failed: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    start_dt = pd.Timestamp(START_DATE, tz='UTC')
    end_dt = pd.Timestamp(END_DATE, tz='UTC')
    df = df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

    df.to_pickle(DATA_FILE)
    return df

def download_model():
    print(f"\n[MODEL] Downloading {MODEL_FILENAME} from Hugging Face...")
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{HF_FOLDER}/{MODEL_FILENAME}", local_dir=".")
        actual_path = os.path.join(".", HF_FOLDER, MODEL_FILENAME)
        if not os.path.exists(actual_path): actual_path = model_path
        return actual_path
    except Exception as e:
        print(f"[ERROR] Model download failed: {e}")
        sys.exit(1)

def get_grid_indices(df, step_size):
    # Standard training logic (cumprod)
    close_array = df['close'].to_numpy()
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    abs_price_log = np.log(abs_price_raw)
    return np.floor(abs_price_log / step_size).astype(int)

def run_inference(df, model_path):
    print(f"\n[INFERENCE] Loading model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    configs = model_data['ensemble_configs']
    total_len = len(df)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    # Pre-calc grids
    config_grids = []
    for cfg in configs:
        grid = get_grid_indices(df, cfg['step_size'])
        max_lookback = cfg['seq_len']
        slice_start = idx_80 - max_lookback
        val_slice = grid[slice_start:idx_90]
        config_grids.append({'cfg': cfg, 'val_seq': val_slice, 'offset': slice_start})

    correct, total_trades = 0, 0
    range_len = idx_90 - idx_80
    
    print(f"[INFERENCE] Testing on {range_len} steps...")
    
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

def run_live_prediction(anchor_price, model_path):
    print(f"\n{'='*40}")
    print(f"LIVE PREDICTION (ETH/USDT)")
    print(f"{'='*40}")
    
    # 1. Load Model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    configs = model_data['ensemble_configs']
    
    # 2. Fetch Latest Data (Small batch only)
    print(f"[LIVE] Anchor Price (Jan 2020): {anchor_price:.2f}")
    print("[LIVE] Fetching last 50 candles...")
    exchange = ccxt.binance({'enableRateLimit': True})
    live_ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=50)
    
    if not live_ohlcv:
        print("[ERROR] Could not fetch live data.")
        return

    # Process: DataFrame -> Remove unfinished -> Last 10
    live_df = pd.DataFrame(live_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms', utc=True)
    live_df = live_df.iloc[:-1] # Drop unfinished
    
    # 3. DIRECT CALCULATION (No stitching)
    # Formula: floor( log(Price_t / Anchor) / step )
    prices = live_df['close'].to_numpy()
    log_prices = np.log(prices / anchor_price)
    
    print("\n--- Last 5 Finished Candles ---")
    print(live_df.tail(5)[['timestamp', 'close']].to_string(index=False))
    print("--------------------------------")

    up_votes = 0
    down_votes = 0
    
    print("\n[LIVE] Computing votes...")
    for cfg in configs:
        step_size = cfg['step_size']
        seq_len = cfg['seq_len']
        
        # Calculate grid indices directly
        grid = np.floor(log_prices / step_size).astype(int)
        
        if len(grid) < seq_len: continue
            
        # Extract sequence (last seq_len candles)
        current_seq_tuple = tuple(grid[-seq_len:])
        current_level = current_seq_tuple[-1]
        
        if current_seq_tuple in cfg['patterns']:
            history = cfg['patterns'][current_seq_tuple]
            predicted_level = Counter(history).most_common(1)[0][0]
            diff = predicted_level - current_level
            
            if diff > 0: up_votes += 1
            elif diff < 0: down_votes += 1
    
    print(f"[LIVE] Votes: UP={up_votes}, DOWN={down_votes}, Configs={len(configs)}")
    if up_votes > 0 and down_votes == 0:
        print(f"\n>>> PREDICTION: LONG (UP) 🟢")
    elif down_votes > 0 and up_votes == 0:
        print(f"\n>>> PREDICTION: SHORT (DOWN) 🔴")
    else:
        print(f"\n>>> PREDICTION: NEUTRAL ⚪")
    print(f"{'='*40}\n")

def main():
    # 1. Fetch/Load History
    df = fetch_or_load_data()
    if df.empty: sys.exit(1)
        
    # ** KEY STEP: Get Anchor Price **
    anchor_price = df['close'].iloc[0]

    # 2. Download Model
    model_path = download_model()

    # 3. Validation (Backtest)
    acc, trades = run_inference(df, model_path)
    
    print(f"\nAccuracy: {acc:.2f}% | Trades: {trades}")
    if (TARGET_ACCURACY - ACCURACY_TOLERANCE) <= acc and (TARGET_TRADES - TRADES_TOLERANCE) <= trades <= (TARGET_TRADES + TRADES_TOLERANCE):
        print("STATUS: PASS ✅")
    else:
        print("STATUS: FAIL ❌")
        
    # 4. Live Prediction (Optimized)
    run_live_prediction(anchor_price, model_path)

if __name__ == "__main__":
    main()

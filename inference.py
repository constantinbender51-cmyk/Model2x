import ccxt
import pandas as pd
import numpy as np
import pickle
import sys
import time
from datetime import datetime
from collections import Counter

# --- CONFIGURATION ---
MODEL_FILENAME = "/app/data/eth.pkl"
SYMBOL = 'ETH/USDT'
TIMEFRAME = '30m'
# Ensure dates match the training period to maintain the 80/90 split logic
START_DATE = '2020-01-01T00:00:00Z'
END_DATE = '2026-01-01T00:00:00Z'

def fetch_binance_data(symbol=SYMBOL, timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    print(f"--- Fetching Data for {symbol} ---")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(start_date)
    end_ts = exchange.parse8601(end_date)
    all_ohlcv = []
    
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Fetched up to {datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=None)}...", end='\r')
            if since >= end_ts: break
            time.sleep(exchange.rateLimit / 1000 * 1.1)
        except Exception as e:
            print(f"\nError: {e}")
            break
            
    print(f"\nTotal rows fetched: {len(all_ohlcv)}")
    if not all_ohlcv: return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df.reset_index(drop=True)

def get_grid_indices(df, step_size):
    close_array = df['close'].to_numpy()
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    abs_price_log = np.log(abs_price_raw)
    return np.floor(abs_price_log / step_size).astype(int)

def run_inference():
    # 1. Load Model
    try:
        with open(MODEL_FILENAME, 'rb') as f:
            model_data = pickle.load(f)
        print(f"\n[LOADED] Model: {MODEL_FILENAME}")
        print(f"Created: {model_data.get('timestamp')}")
        print(f"Ensemble Size: {len(model_data['ensemble_configs'])} configs")
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        sys.exit(1)

    # 2. Fetch Data
    df = fetch_binance_data()
    if df.empty: sys.exit(1)

    # 3. Define Validation Range (80% -> 90%)
    total_len = len(df)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    print(f"\n--- TESTING ON RANGE 80%-90% ---")
    print(f"Time Range: {df['timestamp'].iloc[idx_80]} -> {df['timestamp'].iloc[idx_90]}")

    ensemble_configs = model_data['ensemble_configs']
    
    # Pre-calculate grid sequences for each config in the ensemble
    processed_configs = []
    for cfg in ensemble_configs:
        full_grid = get_grid_indices(df, cfg['step_size'])
        processed_configs.append({
            'cfg': cfg,
            'val_slice': full_grid[idx_80:idx_90]
        })

    # 4. Simulation Logic
    # We must find the max sequence length to know where to safely start
    max_s_len = max(cfg['seq_len'] for cfg in ensemble_configs)
    slice_len = idx_90 - idx_80
    
    combined_correct = 0
    combined_total = 0
    
    print(f"Running simulation (Starting at offset {max_s_len})...")
    
    # Iterate through time by TARGET index
    for target_idx in range(max_s_len, slice_len):
        up_votes = []
        down_votes = []
        
        for item in processed_configs:
            cfg = item['cfg']
            val_seq = item['val_slice']
            s_len = cfg['seq_len']
            
            # Extract sequence ending just before target_idx
            current_seq = tuple(val_seq[target_idx - s_len : target_idx])
            current_level = current_seq[-1]
            
            if current_seq in cfg['patterns']:
                history = cfg['patterns'][current_seq]
                predicted_level = Counter(history).most_common(1)[0][0]
                diff = predicted_level - current_level
                
                if diff > 0: up_votes.append(item)
                elif diff < 0: down_votes.append(item)
        
        # 5. Ensemble Voting (Unanimity logic from training script)
        decision = "HOLD"
        voters = []
        
        if len(up_votes) > 0 and len(down_votes) == 0:
            decision = "UP"
            voters = up_votes
        elif len(down_votes) > 0 and len(up_votes) == 0:
            decision = "DOWN"
            voters = down_votes
            
        # 6. Verification
        if decision != "HOLD":
            # Select the config with the largest step_size for verification (per training logic)
            best_voter = max(voters, key=lambda x: x['cfg']['step_size'])
            best_seq = best_voter['val_slice']
            
            actual_diff = best_seq[target_idx] - best_seq[target_idx - 1]
            
            if actual_diff != 0:
                combined_total += 1
                if (decision == "UP" and actual_diff > 0) or (decision == "DOWN" and actual_diff < 0):
                    combined_correct += 1

    # 7. Final Results
    acc = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
    print("-" * 35)
    print(f"Total Trades: {combined_total}")
    print(f"Accuracy:     {acc:.2f}%")
    print("-" * 35)

if __name__ == "__main__":
    run_inference()

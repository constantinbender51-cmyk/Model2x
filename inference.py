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

# --- CONFIGURATION ---
# Must match training exactly for index alignment
SYMBOL = 'ETH/USDT'
TIMEFRAME = '30m'
START_DATE = '2020-01-01T00:00:00Z'
END_DATE = '2026-01-01T00:00:00Z'

# Paths
DATA_DIR = "/app/data"
DATA_FILE = os.path.join(DATA_DIR, "ohlcv_data.pkl")
MODEL_FILENAME = "eth.pkl"

# Hugging Face Config (Repo to download from)
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"

# Pass Criteria
TARGET_ACCURACY = 85.0  # Approx ~85%
ACCURACY_TOLERANCE = 5.0 # Allow 80-90 range
TARGET_TRADES = 1100
TRADES_TOLERANCE = 100

def ensure_directories():
    if not os.path.exists(DATA_DIR):
        print(f"Creating directory: {DATA_DIR}")
        os.makedirs(DATA_DIR)

def fetch_or_load_data():
    ensure_directories()
    
    # 1. Try Loading Local
    if os.path.exists(DATA_FILE):
        print(f"[DATA] Found local data at {DATA_FILE}. Loading...")
        try:
            df = pd.read_pickle(DATA_FILE)
            print(f"[DATA] Loaded {len(df)} rows.")
            return df
        except Exception as e:
            print(f"[DATA] Error loading pickle: {e}. Re-fetching.")

    # 2. Fetch if not found
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

    print(f"\n[DATA] Fetch complete. Rows: {len(all_ohlcv)}")
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Filter exact range
    start_dt = pd.Timestamp(START_DATE, tz='UTC')
    end_dt = pd.Timestamp(END_DATE, tz='UTC')
    df = df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

    # Save
    print(f"[DATA] Saving to {DATA_FILE}...")
    df.to_pickle(DATA_FILE)
    return df

def download_model():
    print(f"\n[MODEL] Downloading {MODEL_FILENAME} from Hugging Face ({HF_REPO_ID})...")
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"{HF_FOLDER}/{MODEL_FILENAME}",
            local_dir="." # Download to current directory
        )
        # Move it to current dir if nested by hf_hub_download logic, 
        # but usually local_dir handles it. We assume it's accessible.
        # hf_hub_download might create the folder structure locally.
        # We need to find the actual file if it placed it in a subdir.
        
        actual_path = os.path.join(".", HF_FOLDER, MODEL_FILENAME)
        if not os.path.exists(actual_path):
            # If download flattened it or put it elsewhere, try standard name
            actual_path = model_path
            
        print(f"[MODEL] Downloaded to: {actual_path}")
        return actual_path
    except Exception as e:
        print(f"[ERROR] Model download failed: {e}")
        sys.exit(1)

def get_grid_indices(df, step_size):
    # Exact reproduction of training logic
    close_array = df['close'].to_numpy()
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    abs_price_log = np.log(abs_price_raw)
    grid_indices = np.floor(abs_price_log / step_size).astype(int)
    return grid_indices

def run_inference(df, model_path):
    print(f"\n[INFERENCE] Loading model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    configs = model_data['ensemble_configs']
    print(f"[INFERENCE] Loaded ensemble with {len(configs)} configurations.")

    # --- RECONSTRUCT VALIDATION SET (80% - 90%) ---
    # We must calculate grid indices for the *whole* DF first to maintain consistency,
    # then slice the indices, exactly as the training script did.
    
    total_len = len(df)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    print(f"[INFERENCE] Total Data: {total_len} rows.")
    print(f"[INFERENCE] Testing on Validation Slice (80%-90%): Indices {idx_80} to {idx_90}")
    
    # Pre-calculate grid indices for all configs to save time
    # config_grids[i] = full array of grid indices for config i
    config_grids = []
    for cfg in configs:
        grid = get_grid_indices(df, cfg['step_size'])
        # Store only the validation slice + lookback buffer
        # We need enough history before idx_80 to form the first sequence
        max_lookback = cfg['seq_len']
        # We slice from (idx_80 - max_lookback) to idx_90
        slice_start = idx_80 - max_lookback
        val_slice = grid[slice_start:idx_90]
        config_grids.append({
            'cfg': cfg,
            'val_seq': val_slice, 
            'offset': slice_start # To map back to relative indices
        })

    correct = 0
    total_trades = 0
    
    # Iterate through the validation period
    # We iterate i from 0 to (idx_90 - idx_80). 
    # The 'target' index in the FULL array is idx_80 + i.
    range_len = idx_90 - idx_80
    
    print(f"[INFERENCE] Running ensemble voting on {range_len} time steps...")
    
    for i in range(range_len):
        target_abs_idx = idx_80 + i
        
        up_votes = 0
        down_votes = 0
        
        valid_voters = [] # To store who voted what for tie-breaking/verification
        
        for grid_data in config_grids:
            cfg = grid_data['cfg']
            val_seq = grid_data['val_seq']
            offset = grid_data['offset']
            seq_len = cfg['seq_len']
            
            # The index of the target in our 'val_seq' slice
            local_target_idx = target_abs_idx - offset
            
            # We need history [local_target_idx - seq_len : local_target_idx]
            # to predict local_target_idx
            if local_target_idx < seq_len:
                continue # Not enough data yet (shouldn't happen with our buffer, but safety first)
                
            current_seq_tuple = tuple(val_seq[local_target_idx - seq_len : local_target_idx])
            current_level = current_seq_tuple[-1]
            
            if current_seq_tuple in cfg['patterns']:
                history = cfg['patterns'][current_seq_tuple]
                # Majority Vote
                predicted_level = Counter(history).most_common(1)[0][0]
                diff = predicted_level - current_level
                
                if diff > 0:
                    up_votes += 1
                    valid_voters.append(('up', cfg, val_seq, local_target_idx))
                elif diff < 0:
                    down_votes += 1
                    valid_voters.append(('down', cfg, val_seq, local_target_idx))

        # --- ENSEMBLE LOGIC (UNANIMITY) ---
        prediction = 0 # 0: None, 1: Long, -1: Short
        
        if up_votes > 0 and down_votes == 0:
            prediction = 1
            # "Best" config logic (Highest step size)
            best_voter = max([v for v in valid_voters if v[0] == 'up'], key=lambda x: x[1]['step_size'])
        elif down_votes > 0 and up_votes == 0:
            prediction = -1
            best_voter = max([v for v in valid_voters if v[0] == 'down'], key=lambda x: x[1]['step_size'])
            
        if prediction != 0:
            # Check correctness
            # We use the grid of the "best" voter to determine ground truth movement
            _, _, chosen_seq, local_idx = best_voter
            
            curr_lvl = chosen_seq[local_idx - 1]
            actual_next = chosen_seq[local_idx]
            actual_diff = actual_next - curr_lvl
            
            if actual_diff != 0:
                total_trades += 1
                if (prediction == 1 and actual_diff > 0) or (prediction == -1 and actual_diff < 0):
                    correct += 1

    accuracy = (correct / total_trades * 100) if total_trades > 0 else 0.0
    return accuracy, total_trades

def main():
    # 1. Fetch Data
    df = fetch_or_load_data()
    if df.empty:
        print("Data is empty. Exiting.")
        sys.exit(1)

    # 2. Download Model
    model_path = download_model()

    # 3. Run Inference
    acc, trades = run_inference(df, model_path)

    # 4. Evaluate Pass/Fail
    print(f"\n{'='*40}")
    print(f"RESULTS ON 80-90% SPLIT (2020-2026)")
    print(f"{'='*40}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Trades:   {trades}")
    print(f"{'-'*40}")
    
    # Criteria Check
    # Accuracy: ~85% (We allow a small buffer, e.g., > 80% to be generous, or strict 84-86)
    # Trades: 1100 +- 100 (1000 to 1200)
    
    acc_pass = (TARGET_ACCURACY - ACCURACY_TOLERANCE) <= acc # >= 80% essentially
    trades_pass = (TARGET_TRADES - TRADES_TOLERANCE) <= trades <= (TARGET_TRADES + TRADES_TOLERANCE)
    
    if acc_pass and trades_pass:
        print("STATUS: PASS ✅")
        print("Metric Logic: Accuracy ~85% and Trades approx 1100.")
    else:
        print("STATUS: FAIL ❌")
        print(f"Reason: {'Accuracy too low ' if not acc_pass else ''}{'Trades out of range' if not trades_pass else ''}")

if __name__ == "__main__":
    main()

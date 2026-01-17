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

# Hugging Face Config
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"

# Pass Criteria
TARGET_ACCURACY = 85.0
ACCURACY_TOLERANCE = 5.0
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
            local_dir="." 
        )
        actual_path = os.path.join(".", HF_FOLDER, MODEL_FILENAME)
        if not os.path.exists(actual_path):
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

    total_len = len(df)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    print(f"[INFERENCE] Total Data: {total_len} rows.")
    print(f"[INFERENCE] Testing on Validation Slice (80%-90%): Indices {idx_80} to {idx_90}")
    
    config_grids = []
    for cfg in configs:
        grid = get_grid_indices(df, cfg['step_size'])
        max_lookback = cfg['seq_len']
        slice_start = idx_80 - max_lookback
        val_slice = grid[slice_start:idx_90]
        config_grids.append({
            'cfg': cfg,
            'val_seq': val_slice, 
            'offset': slice_start 
        })

    correct = 0
    total_trades = 0
    range_len = idx_90 - idx_80
    
    print(f"[INFERENCE] Running ensemble voting on {range_len} time steps...")
    
    for i in range(range_len):
        target_abs_idx = idx_80 + i
        up_votes = 0
        down_votes = 0
        valid_voters = [] 
        
        for grid_data in config_grids:
            cfg = grid_data['cfg']
            val_seq = grid_data['val_seq']
            offset = grid_data['offset']
            seq_len = cfg['seq_len']
            local_target_idx = target_abs_idx - offset
            
            if local_target_idx < seq_len:
                continue 
                
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
            curr_lvl = chosen_seq[local_idx - 1]
            actual_next = chosen_seq[local_idx]
            actual_diff = actual_next - curr_lvl
            
            if actual_diff != 0:
                total_trades += 1
                if (prediction == 1 and actual_diff > 0) or (prediction == -1 and actual_diff < 0):
                    correct += 1

    accuracy = (correct / total_trades * 100) if total_trades > 0 else 0.0
    return accuracy, total_trades

def run_live_prediction(history_df, model_path):
    print(f"\n{'='*40}")
    print(f"LIVE PREDICTION (ETH/USDT)")
    print(f"{'='*40}")
    
    # 1. Load Model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    configs = model_data['ensemble_configs']
    
    # 2. Fetch Latest Data
    print("[LIVE] Fetching most recent candles from Binance...")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Fetch a buffer (e.g. 50) to ensure we can stitch correctly
    # We only display the last 10, but we need more context to avoid gaps if possible
    live_ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=50)
    
    if not live_ohlcv:
        print("[ERROR] Could not fetch live data.")
        return

    # 3. Process Live Data
    # Convert to DF
    live_df = pd.DataFrame(live_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms', utc=True)
    
    # Remove unfinished candle (Last one returned by Binance is usually current/incomplete)
    # We check if the last timestamp matches the current time roughly, but standard practice 
    # for 'exclude unfinished' is just to drop the last row if using standard fetch_ohlcv
    live_df = live_df.iloc[:-1] 
    
    # 4. Stitch with History (Crucial for Grid Alignment)
    # The Grid Index calculation is path-dependent (cumprod). 
    # We must append live data to the end of history_df to get correct integer levels.
    full_df = pd.concat([history_df, live_df]).drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    print(f"[LIVE] Combined History: {len(history_df)} rows + New Live Data. Total: {len(full_df)}")
    
    # 5. Display Last 10 Candles
    print("\n--- Last 10 Finished Candles ---")
    recent_10 = full_df.tail(10)[['timestamp', 'close', 'volume']]
    print(recent_10.to_string(index=False))
    print("--------------------------------")

    # 6. Run Ensemble Prediction on the VERY LAST step
    up_votes = 0
    down_votes = 0
    
    # We need to calculate the grid for the *entire* stitched series
    # then look at the last point.
    
    print("\n[LIVE] Computing votes...")
    
    for cfg in configs:
        # Recalculate grid on full path
        grid = get_grid_indices(full_df, cfg['step_size'])
        seq_len = cfg['seq_len']
        
        # We need the sequence ending at the last finished candle
        # grid[-1] is the level of the last finished candle.
        # sequence is grid[-(seq_len):] -> This includes the current level as the last item
        # Wait, training logic:
        # current_seq_tuple = tuple(val_seq[local_target_idx - seq_len : local_target_idx])
        # This means the tuple DOES NOT include the target. It includes the history leading up to it.
        # So for live prediction of NEXT move, we take the last `seq_len` indices available.
        
        if len(grid) < seq_len:
            continue
            
        current_seq_tuple = tuple(grid[-seq_len:])
        current_level = current_seq_tuple[-1] # The level we are sitting at right now
        
        if current_seq_tuple in cfg['patterns']:
            history = cfg['patterns'][current_seq_tuple]
            predicted_level = Counter(history).most_common(1)[0][0]
            diff = predicted_level - current_level
            
            if diff > 0:
                up_votes += 1
            elif diff < 0:
                down_votes += 1
    
    # 7. Final Decision
    print(f"[LIVE] Votes: UP={up_votes}, DOWN={down_votes}, Total Configs={len(configs)}")
    
    if up_votes > 0 and down_votes == 0:
        decision = "LONG (UP) 🟢"
    elif down_votes > 0 and up_votes == 0:
        decision = "SHORT (DOWN) 🔴"
    else:
        decision = "NEUTRAL / NO TRADE ⚪" # Mixed votes or no matches
        
    print(f"\n>>> PREDICTION: {decision}")
    print(f"{'='*40}\n")

def main():
    # 1. Fetch Data
    df = fetch_or_load_data()
    if df.empty:
        print("Data is empty. Exiting.")
        sys.exit(1)

    # 2. Download Model
    model_path = download_model()

    # 3. Run Inference (Backtest)
    acc, trades = run_inference(df, model_path)

    # 4. Evaluate Pass/Fail
    print(f"\n{'='*40}")
    print(f"RESULTS ON 80-90% SPLIT")
    print(f"{'='*40}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Trades:   {trades}")
    print(f"{'-'*40}")
    
    acc_pass = (TARGET_ACCURACY - ACCURACY_TOLERANCE) <= acc 
    trades_pass = (TARGET_TRADES - TRADES_TOLERANCE) <= trades <= (TARGET_TRADES + TRADES_TOLERANCE)
    
    if acc_pass and trades_pass:
        print("STATUS: PASS ✅")
    else:
        print("STATUS: FAIL ❌")
        
    # 5. Run Live Prediction
    # We pass the 'df' which contains the 2020-2026 history.
    # The function will append the latest 2026+ data to it.
    run_live_prediction(df, model_path)

if __name__ == "__main__":
    main()

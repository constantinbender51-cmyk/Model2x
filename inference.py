import os
import sys
import pickle
import time
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timezone
from collections import Counter
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
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

def run_live_prediction(anchor_price, model_data):
    """
    Runs a single prediction based on the most recently closed candle.
    Accepts loaded model_data to avoid reloading pickle every 30m.
    """
    print(f"\n{'='*40}")
    print(f"LIVE PREDICTION (ETH/USDT) - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*40}")
    
    configs = model_data['ensemble_configs']
    
    # Fetch Data
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        live_ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=50)
    except Exception as e:
        print(f"[ERROR] Could not fetch live data: {e}")
        return

    if not live_ohlcv:
        print("[ERROR] No data returned from exchange.")
        return

    live_df = pd.DataFrame(live_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms', utc=True)
    
    # IMPORTANT: Drop the last row as it is the currently forming (unfinished) candle
    # We predict based on the 'close' of the previous completed candle.
    live_df = live_df.iloc[:-1] 
    
    prices = live_df['close'].to_numpy()
    log_prices = np.log(prices / anchor_price)
    
    last_candle_time = live_df.iloc[-1]['timestamp']
    print(f"Latest CLOSED Candle: {last_candle_time} | Price: {prices[-1]:.2f}")

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
    
    print(f"[LIVE] Votes: UP={up_votes}, DOWN={down_votes}, Configs={len(configs)}")
    if up_votes > 0 and down_votes == 0:
        print(f"\n>>> PREDICTION: LONG (UP) 🟢")
    elif down_votes > 0 and up_votes == 0:
        print(f"\n>>> PREDICTION: SHORT (DOWN) 🔴")
    else:
        print(f"\n>>> PREDICTION: NEUTRAL ⚪")
    print(f"{'='*40}\n")

def start_live_bot_loop(anchor_price, model_path):
    print(f"\n[SYSTEM] Initializing Live Bot Loop...")
    
    # Load model once to save IO
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    print(f"[SYSTEM] Model loaded. Anchor Price: {anchor_price:.2f}")
    
    while True:
        try:
            # 1. Run the Prediction
            run_live_prediction(anchor_price, model_data)
            
            # 2. Calculate Sleep Time until next 30m candle close
            # Candles close at :00 and :30
            now = time.time()
            interval = 30 * 60 # 1800 seconds
            
            # Calculate next interval timestamp
            next_timestamp = ((now // interval) + 1) * interval
            
            # Sleep until next timestamp + 10 seconds buffer (to let API index the candle)
            sleep_duration = next_timestamp - now + 10
            
            next_run_dt = datetime.fromtimestamp(next_timestamp + 10)
            print(f"[WAIT] Sleeping {sleep_duration/60:.2f} minutes.")
            print(f"[WAIT] Next run scheduled for: {next_run_dt.strftime('%H:%M:%S')}")
            
            time.sleep(sleep_duration)
            
        except KeyboardInterrupt:
            print("\n[STOP] Bot stopped by user.")
            break
        except Exception as e:
            print(f"\n[ERROR] Unexpected error in loop: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

def main():
    # 1. Fetch/Load History to get Anchor Price
    df = fetch_or_load_data()
    if df.empty: sys.exit(1)
        
    anchor_price = df['close'].iloc[0]

    # 2. Download Model
    model_path = download_model()

    # 3. Validation (Backtest) - Only runs once on startup
    acc, trades = run_inference(df, model_path)
    
    print(f"\nAccuracy: {acc:.2f}% | Trades: {trades}")
    if (TARGET_ACCURACY - ACCURACY_TOLERANCE) <= acc and (TARGET_TRADES - TRADES_TOLERANCE) <= trades <= (TARGET_TRADES + TRADES_TOLERANCE):
        print("STATUS: PASS ✅")
    else:
        print("STATUS: FAIL ❌")
        
    # 4. Start the Continuous Loop
    start_live_bot_loop(anchor_price, model_path)

if __name__ == "__main__":
    main()

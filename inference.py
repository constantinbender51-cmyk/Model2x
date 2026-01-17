import ccxt
import pandas as pd
import numpy as np
import pickle
import sys
import time
from datetime import datetime
from collections import Counter

# --- CONFIGURATION ---
MODEL_FILENAME = "grid_ensemble_model.pkl"  # Corrected Filename
SYMBOL = 'ETH/USDT'
TIMEFRAME = '30m'
START_DATE = '2020-01-01T00:00:00Z'
END_DATE = '2026-01-01T00:00:00Z'
SEQ_LENGTH = 5

def fetch_binance_data(symbol=SYMBOL, timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """
    Fetches historical OHLCV data from Binance.
    """
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
    
    # Filter exact date range
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    return df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

def get_grid_indices(df, step_size):
    """
    Converts price series into grid indices based on log-price steps.
    """
    close_array = df['close'].to_numpy()
    # Calculate percentage changes to reconstruct the relative path
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    
    # Reconstruct absolute price path (normalized)
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    abs_price_log = np.log(abs_price_raw)
    
    # Quantize to grid
    grid_indices = np.floor(abs_price_log / step_size).astype(int)
    return grid_indices

def run_inference():
    # 1. Load Model
    try:
        with open(MODEL_FILENAME, 'rb') as f:
            model_data = pickle.load(f)
        print(f"\n[LOADED] Model from {MODEL_FILENAME}")
        print(f"Timestamp: {model_data.get('timestamp', 'Unknown')}")
        print(f"Ensemble Size: {len(model_data['ensemble_configs'])} configurations")
    except FileNotFoundError:
        print(f"[ERROR] {MODEL_FILENAME} not found. Please ensure the model file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        sys.exit(1)

    # 2. Fetch Data
    df = fetch_binance_data()
    if df.empty:
        print("[ERROR] No data fetched.")
        sys.exit(1)

    # 3. Define the Test Range (80% -> 90%)
    total_len = len(df)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    print(f"\n--- TESTING ON RANGE 80%-90% ---")
    print(f"Data Indices: {idx_80} to {idx_90}")
    print(f"Time Range: {df['timestamp'].iloc[idx_80]} -> {df['timestamp'].iloc[idx_90]}")

    ensemble_configs = model_data['ensemble_configs']
    
    # Store the sliced grid sequences for the test period
    test_sequences = []
    
    for cfg in ensemble_configs:
        full_grid = get_grid_indices(df, cfg['step_size'])
        # Extract the validation slice
        val_slice = full_grid[idx_80:idx_90]
        
        test_sequences.append({
            'cfg': cfg,
            'sequence': val_slice
        })

    # 4. Run Ensemble Prediction
    slice_len = idx_90 - idx_80
    
    combined_correct = 0
    combined_total = 0
    
    print("\nRunning simulation...")
    
    for i in range(slice_len - SEQ_LENGTH):
        up_votes = []
        down_votes = []
        
        # Check every model in the ensemble
        for item in test_sequences:
            cfg = item['cfg']
            seq_data = item['sequence']
            
            # Current pattern
            current_seq_tuple = tuple(seq_data[i : i + SEQ_LENGTH])
            current_level = current_seq_tuple[-1]
            
            # Check if this pattern exists in the trained memory
            if current_seq_tuple in cfg['patterns']:
                history = cfg['patterns'][current_seq_tuple]
                # Predict next move based on history
                predicted_next = Counter(history).most_common(1)[0][0]
                diff = predicted_next - current_level
                
                if diff > 0:
                    up_votes.append(item)
                elif diff < 0:
                    down_votes.append(item)
        
        # 5. Ensemble Decision Logic (Consensus)
        if len(up_votes) > 0 and len(down_votes) == 0:
            decision = "UP"
            voters = up_votes
        elif len(down_votes) > 0 and len(up_votes) == 0:
            decision = "DOWN"
            voters = down_votes
        else:
            decision = "HOLD"
            
        # 6. Verify against Reality
        if decision != "HOLD":
            best_voter = max(voters, key=lambda x: x['cfg']['step_size'])
            best_seq = best_voter['sequence']
            
            curr_lvl = best_seq[i + SEQ_LENGTH - 1] # The last known point
            next_lvl = best_seq[i + SEQ_LENGTH]     # The actual next point
            
            actual_diff = next_lvl - curr_lvl
            
            if actual_diff != 0:
                combined_total += 1
                if (decision == "UP" and actual_diff > 0) or (decision == "DOWN" and actual_diff < 0):
                    combined_correct += 1

    # 7. Final Results
    acc = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
    
    print("-" * 30)
    print(f"RESULTS FOR VALIDATION SLICE (80-90%)")
    print("-" * 30)
    print(f"Total Trades Taken: {combined_total}")
    print(f"Correct Trades:     {combined_correct}")
    print(f"Accuracy:           {acc:.2f}%")
    print("-" * 30)
    
    if acc > 50.0:
        print(">>> SUCCESS <<<")
    else:
        print(">>> FAILURE <<<")

if __name__ == "__main__":
    run_inference()

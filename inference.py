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
import traceback
from datetime import datetime, timezone, timedelta
from collections import Counter
from huggingface_hub import hf_hub_download

# --- LOGGING CONFIGURATION ---
# Enhanced format to include filename and line number for precise debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler(sys.stdout)
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
    if not os.path.exists(DATA_DIR):
        logger.info(f"Creating data directory: {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)

# --- ENTRY PRICE TRACKING (NOT IN DB) ---

def load_entry_prices():
    if os.path.exists(ENTRY_TRACKER_FILE):
        try:
            with open(ENTRY_TRACKER_FILE, 'r') as f:
                data = json.load(f)
                logger.debug(f"Loaded entry prices for {len(data)} assets.")
                return data
        except Exception as e:
            logger.error(f"Failed to load entry tracker: {e}")
            return {}
    return {}

def save_entry_prices(tracker):
    try:
        with open(ENTRY_TRACKER_FILE, 'w') as f:
            json.dump(tracker, f)
    except Exception as e:
        logger.error(f"Failed to save entry tracker: {e}")

# --- DATABASE FUNCTIONS ---

def get_db_connection():
    if not DATABASE_URL:
        logger.critical("DATABASE_URL environment variable is missing.")
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def init_db():
    """Initializes tables and renames 'outcome' to 'pnl'"""
    logger.info("Initializing database schema...")
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal_history (
                    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    asset TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    pnl NUMERIC NOT NULL,
                    PRIMARY KEY (time, asset)
                );
            """)
            # Migration check
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
            logger.info("Database initialization and migration successful.")
        except Exception as e:
            logger.critical(f"Database initialization failed: {e}\n{traceback.format_exc()}")
            sys.exit(1) # Critical failure

def settle_and_update_fast(asset, new_pred, p_market, entry_tracker):
    conn = get_db_connection()
    if not conn: 
        logger.error(f"Skipping DB update for {asset}: No connection.")
        return
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT prediction FROM signal WHERE asset = %s", (asset,))
        row = cur.fetchone()
        prev_pred_str = row[0] if row else None
        
        # Settle previous trade if valid
        if prev_pred_str in ["LONG", "SHORT"] and asset in entry_tracker:
            p_entry = entry_tracker[asset]
            p_exit = p_market
            
            if p_entry > 0:
                side = 1.0 if prev_pred_str == "LONG" else -1.0
                pnl = round(side * ((p_exit - p_entry) / p_entry) * 100, 6)
                
                cur.execute("""
                    INSERT INTO signal_history (time, asset, prediction, pnl)
                    VALUES (CURRENT_TIMESTAMP, %s, %s, %s)
                """, (asset, prev_pred_str, pnl))
                conn.commit()
                
                icon = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
                logger.info(f"SETTLE: {asset:<10} | Side: {prev_pred_str:<5} | Entry: {p_entry:.4f} -> Exit: {p_exit:.4f} | PNL: {pnl:+.4f}% {icon}")
            else:
                logger.warning(f"Invalid entry price for {asset} ({p_entry}), skipping settlement calculation.")

        # Insert new signal
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
        conn.commit()
        
        # Update local tracker
        entry_tracker[asset] = p_market
        save_entry_prices(entry_tracker)
        
        log_type = "UPDATE" if prev_pred_str else "INIT"
        logger.info(f"{log_type}: {asset:<10} | New Pred: {new_pred:<8} | Price: {p_market:.4f}")

        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB Transaction failed for {asset}: {e}")

# --- DATA & MODEL FUNCTIONS ---

def get_model_filename(symbol):
    return f"{symbol.split('/')[0].lower()}.pkl"

def download_models():
    logger.info("Starting model download from HuggingFace...")
    model_paths = {}
    success_count = 0
    for symbol in ASSETS:
        fname = get_model_filename(symbol)
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{HF_FOLDER}/{fname}", local_dir=".")
            model_paths[symbol] = path
            success_count += 1
        except Exception as e:
            logger.warning(f"Download failed for {symbol}: {e}")
            continue
    logger.info(f"Downloaded {success_count}/{len(ASSETS)} models.")
    return model_paths

def load_all_models(model_paths_dict):
    logger.info("Loading models into memory...")
    loaded_models = {}
    for symbol, path in model_paths_dict.items():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                loaded_models[symbol] = data
        except Exception as e:
            logger.error(f"Failed to load pickle for {symbol}: {e}")
            continue
    logger.info(f"Successfully loaded {len(loaded_models)} models.")
    return loaded_models

# --- CORE LOGIC ---

def predict_direction(close_prices, anchor_price, configs):
    """
    Pure function to predict direction based on price array.
    """
    prices_arr = np.array(close_prices)
    log_prices = np.log(prices_arr / anchor_price)
    
    up, down = 0, 0
    total_votes = 0
    
    for cfg in configs:
        grid = np.floor(log_prices / cfg['step_size']).astype(int)
        
        if len(grid) < cfg['seq_len']: continue
        
        seq = tuple(grid[-cfg['seq_len']:])
        if seq in cfg['patterns']:
            pred_lvl = Counter(cfg['patterns'][seq]).most_common(1)[0][0]
            if pred_lvl > seq[-1]: up += 1
            elif pred_lvl < seq[-1]: down += 1
            total_votes += 1
            
    # Debug level logging for vote distribution
    # logger.debug(f"Votes: UP={up}, DOWN={down}, Total={total_votes}")
            
    return "LONG" if (up > 0 and down == 0) else "SHORT" if (down > 0 and up == 0) else "NEUTRAL"

# --- VALIDATION LOGIC ---

def validate_asset_accuracy(symbol, anchor_price, model_data, exchange):
    """
    Tests model on the last week of data. Returns accuracy percentage.
    """
    logger.info(f"Validating {symbol} (1-Week Backtest)...")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=500)
        if not ohlcv or len(ohlcv) < 350:
            logger.warning(f"{symbol}: Insufficient history ({len(ohlcv) if ohlcv else 0} candles).")
            return 0.0

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        closes = df['close'].values
        
        window_size = 19
        configs = model_data['ensemble_configs']
        
        correct = 0
        total_trades = 0
        
        # Validation range: Last 336 candles (approx 1 week for 30m candles)
        start_index = len(closes) - 336
        if start_index < window_size: start_index = window_size
        
        logger.debug(f"{symbol}: Testing range index {start_index} to {len(closes)}.")

        for i in range(start_index, len(closes)):
            input_series = closes[i - window_size : i]
            pred = predict_direction(input_series, anchor_price, configs)
            
            prev_close = closes[i-1]
            curr_close = closes[i]
            
            if pred == "LONG":
                total_trades += 1
                if curr_close > prev_close: correct += 1
            elif pred == "SHORT":
                total_trades += 1
                if curr_close < prev_close: correct += 1
                    
        if total_trades == 0:
            logger.warning(f"{symbol}: No trades triggered during validation.")
            return 0.0
            
        accuracy = (correct / total_trades) * 100
        logger.info(f"VALIDATION RESULT [{symbol}]: {correct}/{total_trades} correct | Accuracy: {accuracy:.2f}%")
        return accuracy

    except Exception as e:
        logger.error(f"Validation exception for {symbol}: {e}")
        logger.debug(traceback.format_exc())
        return 0.0

# --- LIVE FUNCTIONS ---

def run_single_asset_live(symbol, anchor_price, model_data, exchange, debug=False):
    try:
        # Fetching slightly more to be safe
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=25) 
        if not ohlcv or len(ohlcv) < 3: 
            logger.warning(f"{symbol}: API returned insufficient data.")
            return "ERROR", None
        
        p_market = ohlcv[-2][4]  # Close of last completed candle
        
        # Inputs: exclude currently forming candle
        close_prices = [x[4] for x in ohlcv[:-1]]
        
        if debug:
            logger.info(f"DEBUG {symbol}: Last 3 closes used for inference: {close_prices[-3:]}")

        res = predict_direction(close_prices, anchor_price, model_data['ensemble_configs'])
        
        return res, p_market
    except Exception as e:
        logger.error(f"Inference error for {symbol}: {e}")
        return "ERROR", None

def start_multi_asset_loop(loaded_models, anchor_prices):
    exchange = ccxt.binance({'enableRateLimit': True})
    entry_tracker = load_entry_prices()
    logger.info(">>> STARTING LIVE TRADING LOOP (Frequency: 30m) <<<")
    
    while True:
        cycle_start = time.time()
        try:
            # Select random assets for verbose debug logging in this cycle
            all_symbols = list(loaded_models.keys())
            debug_symbols = random.sample(all_symbols, min(3, len(all_symbols)))
            logger.info(f"Cycle Debug Targets: {debug_symbols}")

            for symbol, model_data in loaded_models.items():
                anchor = anchor_prices.get(symbol)
                is_debug = symbol in debug_symbols
                
                new_pred, p_market = run_single_asset_live(
                    symbol, 
                    anchor, 
                    model_data, 
                    exchange, 
                    debug=is_debug
                )
                
                if new_pred != "ERROR":
                    settle_and_update_fast(symbol, new_pred, p_market, entry_tracker)
                
                time.sleep(REQUEST_DELAY)

            # Calculation for next 30m mark
            now = time.time()
            # Align to next 30m boundary (e.g., 10:00, 10:30)
            sleep_time = ((now // 1800) + 1) * 1800 - now + 10 
            
            logger.info(f"Cycle completed in {now - cycle_start:.2f}s. Sleeping for {sleep_time/60:.2f} minutes...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("Manual stop detected. Exiting...")
            break
        except Exception as e:
            logger.critical(f"Unexpected Loop Error: {e}")
            logger.critical(traceback.format_exc())
            time.sleep(60)

def main():
    logger.info("--- SYSTEM STARTUP ---")
    ensure_directories()
    init_db()
    
    model_paths = download_models()
    loaded_models = load_all_models(model_paths)
    
    if not loaded_models:
        logger.critical("No models loaded. Shutting down.")
        sys.exit(1)

    available_symbols = list(loaded_models.keys())
    sample_count = min(2, len(available_symbols))
    test_symbols = random.sample(available_symbols, sample_count)
    
    logger.info(f"Performing Pre-Flight Checks on: {test_symbols}")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    validation_passed = True
    
    # --- PRE-FLIGHT CHECKS ---
    for sym in test_symbols:
        anchor = loaded_models[sym]['initial_price']
        
        # 1. Technical Connectivity Check
        pred, p_m = run_single_asset_live(sym, anchor, loaded_models[sym], exchange, debug=True)
        if pred == "ERROR":
            logger.error(f"PRE-FLIGHT FAIL: {sym} connectivity/inference error.")
            validation_passed = False
            break
            
        # 2. Accuracy Check (Hard Constraint > 70%)
        accuracy = validate_asset_accuracy(sym, anchor, loaded_models[sym], exchange)
        if accuracy <= 70.0:
            logger.error(f"PRE-FLIGHT FAIL: {sym} accuracy {accuracy:.2f}% (Threshold: >70%).")
            validation_passed = False
            break
        
        logger.info(f"PRE-FLIGHT SUCCESS: {sym} | Acc: {accuracy:.2f}% | Price: {p_m}")
    # -------------------------

    if validation_passed:
        logger.info("All pre-flight checks passed.")
        anchors = {sym: loaded_models[sym]['initial_price'] for sym in loaded_models}
        start_multi_asset_loop(loaded_models, anchors)
    else:
        logger.critical("Pre-flight checks failed. System halting.")
        sys.exit(1)

if __name__ == "__main__":
    main()

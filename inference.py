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
import logging

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Updated Asset List (matching training script)
ASSETS = [
    # Majors
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 
    'AVAX/USDT', 'DOT/USDT', 'LTC/USDT', 'BCH/USDT',
    # DeFi & Infrastructure
    'LINK/USDT', 'UNI/USDT', 'AAVE/USDT', 'NEAR/USDT', 
    'FIL/USDT', 'ALGO/USDT', 'XLM/USDT', 'EOS/USDT',
    # Meme Coins & Metaverse
    'DOGE/USDT', 'SHIB/USDT', 'SAND/USDT'
]

TIMEFRAME = '30m'
START_DATE = '2020-01-01T00:00:00Z'
END_DATE = '2026-01-01T00:00:00Z'

# Railway / Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# Paths
DATA_DIR = "/app/data"
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"

# Validation Criteria
TARGET_ACCURACY = 70.0  # Matching ensemble threshold from training
ACCURACY_TOLERANCE = 5.0
MIN_TRADES = 50  # Minimum trades needed for validation

# Rate Limiting
API_RETRY_DELAY = 60  # seconds
MAX_RETRIES = 3
REQUEST_DELAY = 0.5  # seconds between requests

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)

# --- DATABASE FUNCTIONS ---

def get_db_connection():
    """Establishes connection to the Railway Postgres DB with error handling"""
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable not found.")
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def init_db():
    """Creates the signal and signal_history tables if they don't exist"""
    logger.info("Initializing database...")
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            
            # Current signals table (keep original structure)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal (
                    asset TEXT PRIMARY KEY,
                    prediction TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP
                );
            """)
            
            # Enhanced signal history table with outcome calculation
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal_history (
                    id SERIAL PRIMARY KEY,
                    asset TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    entry_price NUMERIC NOT NULL,
                    exit_price NUMERIC NOT NULL,
                    return_pct NUMERIC NOT NULL,
                    direction_return NUMERIC NOT NULL,
                    outcome TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indices for querying
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_asset_time 
                ON signal_history (asset, exit_time);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_exit_time 
                ON signal_history (exit_time);
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Tables 'signal' and 'signal_history' are ready.")
        except Exception as e:
            logger.error(f"Failed to init DB: {e}")
            if conn:
                conn.close()

def save_prediction_to_db(asset, prediction, entry_price, start_time, end_time):
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
            
            # Also create a pending history record with entry price
            history_query = """
                INSERT INTO signal_history 
                (asset, prediction, entry_price, entry_time, exit_time)
                VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(history_query, (asset, prediction, entry_price, start_time, end_time))
            
            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"Saved to DB: {asset} -> {prediction} @ {entry_price}")
        except Exception as e:
            logger.error(f"Failed to save prediction for {asset}: {e}")
            if conn:
                conn.close()

def get_last_prediction(asset):
    """
    Gets the last prediction for an asset from the signal table.
    Returns dict with prediction and end_time, or None if not found.
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT prediction, end_time
            FROM signal
            WHERE asset = %s
        """, (asset,))
        
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if row:
            return {
                'prediction': row[0],
                'end_time': row[1]
            }
        return None
        
    except Exception as e:
        logger.error(f"Failed to get last prediction for {asset}: {e}")
        if conn:
            conn.close()
        return None

def process_expired_signals(exchange):
    """
    Check for signals that have expired (end_time <= now) and calculate returns.
    This should be called at the beginning of each 30-minute interval.
    """
    now_utc = datetime.now(timezone.utc)
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        # Get all signals that have just expired
        cur.execute("""
            SELECT s.asset, s.prediction, sh.entry_price, sh.id as history_id
            FROM signal s
            INNER JOIN signal_history sh ON s.asset = sh.asset
            WHERE s.end_time <= %s
            AND sh.exit_price IS NULL
            AND sh.return_pct IS NULL
        """, (now_utc,))
        
        expired_signals = cur.fetchall()
        
        if not expired_signals:
            return
        
        logger.info(f"Processing {len(expired_signals)} expired signals...")
        
        for asset, prediction, entry_price, history_id in expired_signals:
            try:
                # Fetch current price from exchange
                ticker = exchange.fetch_ticker(asset)
                exit_price = ticker['last']
                
                # Calculate raw return percentage
                return_pct = ((exit_price - entry_price) / entry_price) * 100
                
                # Calculate direction-adjusted return
                # LONG: positive return is good, negative is bad
                # SHORT: negative return is good, positive is bad
                # NEUTRAL: 0 return
                direction_multiplier = 0
                if prediction == "LONG":
                    direction_multiplier = 1
                elif prediction == "SHORT":
                    direction_multiplier = -1
                
                direction_return = return_pct * direction_multiplier
                
                # Determine outcome based on direction_return
                if direction_return > 0:
                    outcome = "WIN"
                elif direction_return < 0:
                    outcome = "LOSS"
                else:
                    outcome = "NEUTRAL"
                
                # Update the history record with results
                update_query = """
                    UPDATE signal_history 
                    SET exit_price = %s,
                        return_pct = %s,
                        direction_return = %s,
                        outcome = %s
                    WHERE id = %s
                """
                cur.execute(update_query, (
                    exit_price, 
                    round(return_pct, 4),
                    round(direction_return, 4),
                    outcome,
                    history_id
                ))
                
                # Log the result
                icon = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "➖"
                logger.info(f"Signal closed: {asset} | {prediction} | Entry: {entry_price:.4f} | Exit: {exit_price:.4f} | "
                           f"Return: {return_pct:+.2f}% | Dir. Return: {direction_return:+.2f}% {icon}")
                
                # Rate limiting
                time.sleep(REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"Failed to process expired signal for {asset}: {e}")
                continue
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error in process_expired_signals: {e}")
        if conn:
            conn.close()

def get_performance_stats(days=7):
    """
    Get performance statistics for the last N days.
    Returns: dict with win_rate, total_return, avg_return, etc.
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cur = conn.cursor()
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get all completed trades in the timeframe
        cur.execute("""
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN outcome = 'WIN' THEN 1 END) as wins,
                COUNT(CASE WHEN outcome = 'LOSS' THEN 1 END) as losses,
                AVG(direction_return) as avg_return,
                SUM(direction_return) as total_return,
                STDDEV(direction_return) as std_return
            FROM signal_history
            WHERE exit_time >= %s
            AND outcome IN ('WIN', 'LOSS')
        """, (cutoff_time,))
        
        stats = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if stats and stats[0] > 0:
            total_trades, wins, losses, avg_return, total_return, std_return = stats
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 2),
                'avg_return': round(avg_return or 0, 4),
                'total_return': round(total_return or 0, 4),
                'std_return': round(std_return or 0, 4)
            }
        
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'total_return': 0.0,
            'std_return': 0.0
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        if conn:
            conn.close()
        return None

# --- DATA & MODEL FUNCTIONS ---

def get_model_filename(symbol):
    """Converts symbol (ETH/USDT) to filename (eth.pkl)"""
    return f"{symbol.split('/')[0].lower()}.pkl"

def get_data_filename(symbol):
    """Converts symbol (ETH/USDT) to cached filename"""
    safe_sym = symbol.replace('/', '_')
    return os.path.join(DATA_DIR, f"ohlcv_{safe_sym}.pkl")

def fetch_or_load_data(symbol, max_retries=MAX_RETRIES):
    """
    Fetch or load historical data with retry logic
    """
    ensure_directories()
    data_file = get_data_filename(symbol)
    
    # Try loading from cache first
    if os.path.exists(data_file):
        logger.info(f"Loading cached data for {symbol}")
        try:
            df = pd.read_pickle(data_file)
            # Validate data
            if not df.empty and 'close' in df.columns:
                return df
            logger.warning(f"Cached data for {symbol} is invalid. Re-fetching.")
        except Exception as e:
            logger.warning(f"Error loading cache for {symbol}: {e}. Re-fetching.")

    # Fetch new data
    logger.info(f"Fetching data for {symbol} from Binance")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(START_DATE)
    end_ts = exchange.parse8601(END_DATE)
    all_ohlcv = []
    retries = 0

    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            if since >= end_ts:
                break
                
            time.sleep(exchange.rateLimit / 1000 * 1.1)
            retries = 0  # Reset on success
            
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
            retries += 1
            if retries >= max_retries:
                logger.error(f"Max retries reached for {symbol}")
                break
            logger.warning(f"Rate limit hit for {symbol}. Waiting {API_RETRY_DELAY}s...")
            time.sleep(API_RETRY_DELAY)
            
        except Exception as e:
            logger.error(f"Fetch error for {symbol}: {e}")
            break

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Filter date range
    start_dt = pd.Timestamp(START_DATE, tz='UTC')
    end_dt = pd.Timestamp(END_DATE, tz='UTC')
    df = df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

    # Save to cache
    try:
        df.to_pickle(data_file)
        logger.info(f"Cached {len(df)} rows for {symbol}")
    except Exception as e:
        logger.warning(f"Could not cache data for {symbol}: {e}")
        
    return df

def download_models():
    """
    Downloads models for ALL configured ASSETS from Hugging Face.
    Returns a dictionary: { 'ETH/USDT': '/path/to/eth.pkl', ... }
    """
    logger.info(f"Downloading models for {len(ASSETS)} assets...")
    model_paths = {}
    
    for symbol in ASSETS:
        fname = get_model_filename(symbol)
        try:
            path = hf_hub_download(
                repo_id=HF_REPO_ID, 
                filename=f"{HF_FOLDER}/{fname}", 
                local_dir="."
            )
            
            # Handle HF directory structure
            actual_path = os.path.join(".", HF_FOLDER, fname)
            if not os.path.exists(actual_path): 
                actual_path = path
                
            model_paths[symbol] = actual_path
            logger.info(f"✓ Downloaded model for {symbol}")
            
        except Exception as e:
            logger.warning(f"Failed to download model for {symbol}: {e}")
            
    return model_paths

def load_all_models(model_paths_dict):
    """
    Loads pickle files into memory with validation.
    Returns: { 'ETH/USDT': model_data_dict, ... }
    """
    loaded_models = {}
    logger.info("Loading model files into memory...")
    
    for symbol, path in model_paths_dict.items():
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            # Validate model structure
            required_keys = ['ensemble_configs', 'initial_price']
            if all(key in data for key in required_keys):
                loaded_models[symbol] = data
                logger.info(f"✓ Loaded {symbol} ({len(data['ensemble_configs'])} configs)")
            else:
                logger.warning(f"Invalid model structure for {symbol}")
                
        except Exception as e:
            logger.error(f"Could not load model for {symbol}: {e}")
            
    return loaded_models

# --- INFERENCE LOGIC ---

def get_grid_indices(df, step_size):
    """Convert price data to grid indices"""
    close_array = df['close'].to_numpy()
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    abs_price_log = np.log(abs_price_raw)
    return np.floor(abs_price_log / step_size).astype(int)

def run_backtest_inference(df, model_data):
    """
    Runs historical validation to confirm model quality.
    Returns accuracy and trade count.
    """
    configs = model_data['ensemble_configs']
    total_len = len(df)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    # Prepare grid sequences for each config
    config_grids = []
    for cfg in configs:
        grid = get_grid_indices(df, cfg['step_size'])
        max_lookback = cfg['seq_len']
        slice_start = max(0, idx_80 - max_lookback)
        val_slice = grid[slice_start:idx_90]
        config_grids.append({
            'cfg': cfg, 
            'val_seq': val_slice, 
            'offset': slice_start
        })

    correct, total_trades = 0, 0
    range_len = idx_90 - idx_80
    
    for i in range(range_len):
        target_abs_idx = idx_80 + i
        up_votes, down_votes = [], []
        
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
                    up_votes.append((cfg, val_seq, local_target_idx))
                elif diff < 0:
                    down_votes.append((cfg, val_seq, local_target_idx))

        # Ensemble decision
        prediction = 0
        if len(up_votes) > 0 and len(down_votes) == 0:
            prediction = 1
            best_voter = max(up_votes, key=lambda x: x[0]['step_size'])
        elif len(down_votes) > 0 and len(up_votes) == 0:
            prediction = -1
            best_voter = max(down_votes, key=lambda x: x[0]['step_size'])
            
        if prediction != 0:
            _, chosen_seq, local_idx = best_voter
            if local_idx < len(chosen_seq):
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
    Returns (prediction_string, current_price, previous_candle_close)
    """
    configs = model_data['ensemble_configs']
    
    try:
        live_ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=50)
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
    except Exception as e:
        logger.error(f"Failed to fetch live data for {symbol}: {e}")
        return "ERROR", None, None

    if not live_ohlcv or len(live_ohlcv) < 2:
        return "ERROR", None, None

    live_df = pd.DataFrame(live_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms', utc=True)
    
    # Drop unfinished candle
    live_df = live_df.iloc[:-1] 
    
    if len(live_df) < 2:
        return "ERROR", None, None
    
    # Get previous candle close (the one we just completed)
    previous_candle_close = live_df.iloc[-1]['close']
    
    prices = live_df['close'].to_numpy()
    log_prices = np.log(prices / anchor_price)
    
    up_votes = 0
    down_votes = 0
    
    for cfg in configs:
        step_size = cfg['step_size']
        seq_len = cfg['seq_len']
        grid = np.floor(log_prices / step_size).astype(int)
        
        if len(grid) < seq_len:
            continue
            
        current_seq_tuple = tuple(grid[-seq_len:])
        current_level = current_seq_tuple[-1]
        
        if current_seq_tuple in cfg['patterns']:
            history = cfg['patterns'][current_seq_tuple]
            predicted_level = Counter(history).most_common(1)[0][0]
            diff = predicted_level - current_level
            
            if diff > 0:
                up_votes += 1
            elif diff < 0:
                down_votes += 1
    
    if up_votes > 0 and down_votes == 0:
        return "LONG", current_price, previous_candle_close
    elif down_votes > 0 and up_votes == 0:
        return "SHORT", current_price, previous_candle_close
    else:
        return "NEUTRAL", current_price, previous_candle_close

def start_multi_asset_loop(loaded_models, anchor_prices):
    """
    Main loop for continuous multi-asset prediction
    """
    logger.info(f"Starting Multi-Asset Live Bot Loop for {len(loaded_models)} assets")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    iteration = 0
    
    while True:
        try:
            iteration += 1
            now_utc = datetime.now(timezone.utc)
            start_time = now_utc
            end_time = now_utc + timedelta(minutes=30)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration} - {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info(f"{'='*60}")
            
            # First, process any expired signals from previous period
            process_expired_signals(exchange)
            
            # Display performance stats from last 7 days
            stats = get_performance_stats(days=7)
            if stats and stats['total_trades'] > 0:
                logger.info(f"Performance (7 days): {stats['total_trades']} trades | "
                           f"Win Rate: {stats['win_rate']}% | "
                           f"Total Return: {stats['total_return']:.2f}%")
            
            predictions_made = 0
            
            for symbol, model_data in loaded_models.items():
                anchor = anchor_prices.get(symbol)
                if not anchor:
                    logger.warning(f"Skipping {symbol} - no anchor price")
                    continue
                
                pred, entry_price, previous_close = run_single_asset_live(symbol, anchor, model_data, exchange)
                
                # Console output with emoji
                icon = "⚪"
                if pred == "LONG":
                    icon = "🟢"
                elif pred == "SHORT":
                    icon = "🔴"
                elif pred == "ERROR":
                    icon = "⚠️"
                
                logger.info(f"{symbol:12} | {pred:8} {icon} | Price: {entry_price:.4f}")
                
                # Save to DB (now includes entry price)
                if pred != "ERROR" and entry_price:
                    save_prediction_to_db(symbol, pred, entry_price, start_time, end_time)
                    predictions_made += 1
                
                # Rate limiting
                time.sleep(REQUEST_DELAY)

            logger.info(f"{'='*60}")
            logger.info(f"Predictions made: {predictions_made}/{len(loaded_models)}")

            # Calculate next run time
            now = time.time()
            interval = 30 * 60  # 30 minutes
            next_timestamp = ((now // interval) + 1) * interval
            sleep_duration = next_timestamp - now + 10  # 10s buffer
            
            next_run_dt = datetime.fromtimestamp(next_timestamp + 10)
            logger.info(f"Next run scheduled: {next_run_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info(f"Sleeping for {sleep_duration/60:.1f} minutes...\n")
            
            time.sleep(sleep_duration)
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            logger.info("Waiting 60s before retry...")
            time.sleep(60)

def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("Multi-Asset Crypto Trading Bot - Inference Engine")
    logger.info("="*60)
    
    # 0. Initialize Database
    init_db()

    # 1. Download Models
    model_paths = download_models()
    if not model_paths:
        logger.error("No models downloaded. Exiting.")
        sys.exit(1)
    logger.info(f"Downloaded {len(model_paths)} model files")

    # 2. Load Models
    loaded_models = load_all_models(model_paths)
    if not loaded_models:
        logger.error("No models loaded successfully. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded {len(loaded_models)} models into memory")

    # 3. Validation & Extract Anchor Prices
    valid_models = {}
    anchor_prices = {}
    
    logger.info("\n" + "="*60)
    logger.info("BACKTEST VALIDATION")
    logger.info("="*60)
    
    for symbol in list(loaded_models.keys()):
        logger.info(f"\nValidating {symbol}...")
        
        # Get anchor price from model
        anchor_price = loaded_models[symbol].get('initial_price')
        if not anchor_price:
            logger.warning(f"No anchor price in model for {symbol}. Skipping.")
            continue
            
        anchor_prices[symbol] = anchor_price
        
        # Fetch historical data for validation
        df = fetch_or_load_data(symbol)
        if df.empty:
            logger.warning(f"No historical data for {symbol}. Skipping.")
            continue
        
        # Run backtest
        try:
            acc, trades = run_backtest_inference(df, loaded_models[symbol])
            
            status = "❌ FAIL"
            if acc >= (TARGET_ACCURACY - ACCURACY_TOLERANCE) and trades >= MIN_TRADES:
                status = "✅ PASS"
                valid_models[symbol] = loaded_models[symbol]
            
            logger.info(f"{symbol:12} | Acc: {acc:5.2f}% | Trades: {trades:4d} | {status}")
            
        except Exception as e:
            logger.error(f"Validation error for {symbol}: {e}")

    logger.info("\n" + "="*60)
    logger.info(f"VALIDATION COMPLETE: {len(valid_models)}/{len(loaded_models)} models passed")
    logger.info("="*60 + "\n")

    if not valid_models:
        logger.error("No valid models passed validation. Exiting.")
        sys.exit(1)
        
    # 4. Start the Live Trading Loop
    start_multi_asset_loop(valid_models, anchor_prices)

if __name__ == "__main__":
    main()
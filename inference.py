import pickle
from pathlib import Path

# 1. Ensure the directory exists
data_dir = Path("/app/data/")
data_dir.mkdir(parents=True, exist_ok=True)

# Define file paths
pkl_path = data_dir / "eth.pkl"
txt_path = data_dir / "file.txt"

# 2. Write to eth.pkl
# Example data: a dictionary representing Ethereum-related info
eth_data = {"symbol": "ETH", "network": "Mainnet", "status": "active"}

try:
    with open(pkl_path, "wb") as pkl_file:
        pickle.dump(eth_data, pkl_file)
    print(f"Created: {pkl_path}")

    # 3. Write to file.txt
    with open(txt_path, "w") as txt_file:
        txt_file.write("This is a sample text file created in /app/data/.")
    print(f"Created: {txt_path}")

except PermissionError:
    print("Error: Permission denied. Please run with 'sudo' to write to /app/.")

import os

# Define the path
path = "/app/data/"

try:
    # exist_ok=True prevents an error if the folder already exists
    os.makedirs(path, exist_ok=True)
    print(f"Successfully ensured directory exists at: {path}")
except PermissionError:
    print("Error: You do not have permission to create this directory.")
except Exception as e:
    print(f"An error occurred: {e}")

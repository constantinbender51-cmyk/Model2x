import os

def scan_directory(start_path='/'):
    print(f"--- Scanning for files starting from: {start_path} ---")
    try:
        # Walk through the directory tree
        for root, dirs, files in os.walk(start_path):
            # Skip hidden directories like .git or system dirs to keep output clean
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                # Construct absolute path
                full_path = os.path.join(root, file)
                print(full_path)
                
    except PermissionError:
        print(f"Permission denied for {start_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Start scanning from the current directory, or change to '/' to scan the entire container
    scan_directory('.') 
    # If you want to scan the app folder specifically, uncomment below:
    # scan_directory('/app')

import os

def scan_disk():
    print(f"{'PATH':<60} | {'SIZE'}")
    print("-" * 75)

    # Folders to ignore (Standard Linux system files)
    # We skip these to find your actual data faster
    ignore_dirs = {
        'proc', 'sys', 'dev', 'lib', 'usr', 'bin', 
        'sbin', 'boot', 'run', 'tmp', 'etc', 'var'
    }

    # Walk through the entire directory tree starting at root
    for root, dirs, files in os.walk("/"):
        # Modify dirs in-place to stop os.walk from entering system folders
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for name in files:
            filepath = os.path.join(root, name)
            try:
                # Get file size to help identify database files or large uploads
                size = os.path.getsize(filepath)
                print(f"{filepath:<60} | {size} bytes")
            except OSError:
                # specific permissions error, skip
                pass

if __name__ == "__main__":
    scan_disk()

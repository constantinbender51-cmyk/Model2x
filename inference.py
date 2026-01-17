import os

target = "/app/data"
cwd = os.getcwd()

print(f"1. Python is running from: {cwd}")
print(f"2. Checking absolute path '{target}'...")

if os.path.exists(target):
    contents = os.listdir(target)
    print(f"   STATUS: Found! It contains {len(contents)} items.")
    print(f"   CONTENTS: {contents}")
    
    # Check if we can actually read it (Permissions check)
    if os.access(target, os.R_OK):
        print("   PERMISSIONS: Read access OK.")
    else:
        print("   PERMISSIONS: BLOCKED (Cannot read). Needs root.")
else:
    print("   STATUS: NOT FOUND. The folder does not exist at this path.")

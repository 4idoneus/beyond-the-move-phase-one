import os
import glob

print("--- DEBUGGING PATHS ---")

# 1. Where am I right now?
current_dir = os.getcwd()
print(f" Current Working Directory: {current_dir}")

# 2. Does the data folder exist here?
target_dir = os.path.join(current_dir, 'data', 'raw')
print(f"Looking for data at: {target_dir}")

if os.path.exists(target_dir):
    print(" The folder exists!")
    
    # 3. What is inside?
    files = os.listdir(target_dir)
    print(f" Files found in folder: {files}")
    
    # 4. Do they match the pattern?
    sgf_files = glob.glob(os.path.join(target_dir, '*.sgf'))
    print(f".sgf files matched: {len(sgf_files)}")
    
    if len(files) > 0 and len(sgf_files) == 0:
        print("  WARNING: Files exist but don't match '*.sgf'. Check file extensions (e.g., .SGF vs .sgf or .sgf.txt)")
else:
    print(" The folder does NOT exist here.")
    print("   -> Try moving your terminal one level up: 'cd ..'")
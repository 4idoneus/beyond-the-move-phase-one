import os
import glob
import numpy as np
import time
import datetime
from sgfmill import sgf, boards
from encoder import GoBoardEncoder

# --- CONFIGURATION ---
RAW_DATA_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
MAX_GAMES_TO_PROCESS = 200000  # Set high enough to process all files in your folder

# Create processed folder if it doesn't exist
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

class GameStateWrapper:
    """
    Bridge class: Connects Sgfmills's board format to our Encoder's expected format.
    """
    def __init__(self, sgfmill_board, color_to_move):
        self.board_size = 19
        self.board = np.zeros((19, 19), dtype=int)
        self.color_to_move = color_to_move
        
        # Convert SGFMill board (None, 'b', 'w') to Integers (0, 1, 2)
        for r in range(19):
            for c in range(19):
                color = sgfmill_board.get(r, c)
                if color == 'b':
                    self.board[r][c] = 1
                elif color == 'w':
                    self.board[r][c] = 2

def get_sgf_content(filepath):
    """
    Robust SGF loader. Tries multiple encodings to handle older/Asian files.
    1. UTF-8 (Standard)
    2. GB18030 (Chinese standard, common on Fox/Tygem)
    3. Latin-1 (Fallback - guarantees byte reading, might garble names but keeps moves intact)
    """
    with open(filepath, 'rb') as f:
        raw_data = f.read()
    
    # 1. Try UTF-8
    try:
        return raw_data.decode('utf-8')
    except UnicodeDecodeError:
        pass 
        
    # 2. Try GB18030
    try:
        return raw_data.decode('gb18030')
    except UnicodeDecodeError:
        pass 

    # 3. Try Latin-1 (The "Nuclear Option" - reads anything as bytes)
    try:
        return raw_data.decode('latin-1')
    except Exception:
        return None

def process_games():
    start_time = time.time()  # Start timing
    print(f"  Process started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    encoder = GoBoardEncoder()
    # Case insensitive search for .sgf, .SGF, etc.
    files = glob.glob(os.path.join(RAW_DATA_DIR, '*.[sS][gG][fF]'))
    
    total_found = len(files)
    print(f" Found {total_found} SGF files. Processing ALL valid games...")
    
    processed_count = 0
    rejected_count = 0
    
    for i, filepath in enumerate(files):
        if processed_count >= MAX_GAMES_TO_PROCESS:
            break

        sgf_content = get_sgf_content(filepath)
        if sgf_content is None:
            rejected_count += 1
            print(f" Critical Read Error (Corrupt File): {os.path.basename(filepath)}")
            continue

        try:
            game = sgf.Sgf_game.from_string(sgf_content)
            
            # --- NO FILTER ---
            # Trust that data/raw contains the valid Pro/9d games.
            # Process ALL parsed games.
            
            board = boards.Board(19)
            filename_base = os.path.basename(filepath).replace('.sgf', '')
            
            # Temporary lists to hold all moves for THIS game
            game_inputs = []
            game_targets = []
            
            # Replay the game move by move
            for move_node in game.get_main_sequence():
                color, move_coords = move_node.get_move()
                
                if move_coords is None:
                    continue # Skip pass moves
                
                row, col = move_coords
                
                # 1. ENCODE INPUT (The board BEFORE the move)
                game_state = GameStateWrapper(board, color)
                input_tensor = encoder.encode(game_state)
                
                # 2. ENCODE TARGET (The move that was played)
                # Flatten 19x19 coordinate to a single integer (0-360)
                target_index = row * 19 + col
                
                # Collect into list (Do not save yet)
                game_inputs.append(input_tensor)
                game_targets.append(target_index)
                
                # 3. UPDATE BOARD (Play the move for the next step)
                board.play(row, col, color)
            
            # 4. SAVE ONCE PER GAME (Batch saving)
            # Only save if we actually found moves
            if len(game_inputs) > 0:
                save_path = os.path.join(PROCESSED_DIR, f"{filename_base}.npz")
                
                # Stack lists into numpy arrays
                # Inputs shape: (N_moves, 17, 19, 19)
                # Targets shape: (N_moves,)
                np.savez_compressed(
                    save_path, 
                    inputs=np.array(game_inputs, dtype=np.float32), 
                    targets=np.array(game_targets, dtype=np.int64)
                )
                
                processed_count += 1
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f" Processed {processed_count} / {total_found} |  Total Rejected: {rejected_count} | Elapsed: {elapsed/60:.1f} min")
            else:
                # File was parsed but had no valid moves (e.g. empty game or only pass moves)
                rejected_count += 1

        except Exception as e:
            # If a specific file is corrupted or logic fails, count as rejected
            rejected_count += 1
            # Uncomment below to see specific errors if needed
            # print(f"⚠️ Logic Error in {os.path.basename(filepath)}: {e}")
            continue

    # --- FINAL REPORT ---
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    print("\n" + "="*40)
    print(f" BATCH PROCESSING COMPLETE")
    print("="*40)
    print(f" Total Files Found:   {total_found}")
    print(f" Successfully Saved:  {processed_count}")
    print(f" Rejected/Corrupted:  {rejected_count}")
    if total_found > 0:
        print(f" Success Rate:        {(processed_count/total_found)*100:.1f}%")
    print(f" Total Time:          {hours}h {minutes}m {seconds}s")
    print("="*40)

if __name__ == "__main__":
    process_games()
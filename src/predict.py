import torch
import numpy as np
from src.model import GoResNet
from src.encoder import GoBoardEncoder

# --- CONFIG ---
MODEL_PATH = "models/resnet_go_final.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL = None

class SimpleGameState:
    def __init__(self, board_numpy, color):
        self.board = board_numpy
        self.color_to_move = color

def load_model():
    global _MODEL
    if _MODEL is None:
        try:
            model = GoResNet()
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            _MODEL = model
        except Exception as e:
            return None
    return _MODEL

def idx_to_coord(arg1, arg2=None):
    """ 
    Converts internal coordinates to 'Q16' format.
    Corrected for SGFMill where (0,0) is Bottom-Left.
    """
    if arg2 is not None:
        row, col = arg1, arg2
    else:
        idx = arg1
        row = idx // 19
        col = idx % 19

    # Go columns: A-T (skipping I)
    letters = "ABCDEFGHJKLMNOPQRST"
    
    if 0 <= col < 19 and 0 <= row < 19:
        col_str = letters[col]
        # CRITICAL FIX: SGFMill Row 0 is Bottom (Line 1).
        # So we just add 1 to the index.
        row_str = str(row + 1)
        return f"{col_str}{row_str}"
    return "Pass"

def get_best_moves(board_numpy, player_color, top_k=5):
    model = load_model()
    if model is None: return []

    color_code = 'b' if player_color == 'Black' else 'w'
    game_state = SimpleGameState(board_numpy, color_code)
    
    encoder = GoBoardEncoder()
    input_tensor = encoder.encode(game_state)
    tensor_batch = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor_batch)
        probs = torch.exp(output)
        top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for i in range(top_k):
        idx = top_indices[0][i].item()
        score = top_probs[0][i].item()
        
        # Consistent decoding
        row = idx // 19
        col = idx % 19
        
        results.append({
            "move": idx_to_coord(row, col),
            "confidence": score,
            "row": row,
            "col": col
        })
        
    return results

def predict_opponent_response(board_numpy, my_move_row, my_move_col, my_color):
    next_board = board_numpy.copy()
    stone_val = 1 if my_color == 'Black' else 2
    next_board[my_move_row][my_move_col] = stone_val
    
    opp_color = 'White' if my_color == 'Black' else 'Black'
    opp_moves = get_best_moves(next_board, opp_color, top_k=1)
    
    if opp_moves:
        return opp_moves[0]['move']
    return "Unknown"
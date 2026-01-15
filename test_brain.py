import numpy as np
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import get_best_moves, load_model

# 1. Initialize an Empty Board (19x19 filled with 0s)
empty_board = np.zeros((19, 19), dtype=int)

# 2. Add a few stones manually to simulate a game
# Black at Star Point (Q16 -> Row 3, Col 15)
empty_board[3][15] = 1 
# White approaches (O17 -> Row 2, Col 13)
empty_board[2][13] = 2 

print("🧠 Loading AI Model...")
model = load_model()

if model:
    print("✅ Model Loaded!")
    print("🔮 Asking for Black's next move...")
    
    # 3. Get Prediction
    suggestions = get_best_moves(empty_board, player_color='Black', top_k=3)
    
    for i, s in enumerate(suggestions):
        print(f"   {i+1}. Move: {s['move']} | Confidence: {s['confidence']*100:.1f}%")
else:
    print("❌ Model failed to load. Check 'models/' folder.")
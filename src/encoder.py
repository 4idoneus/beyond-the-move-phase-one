import numpy as np
from sgfmill import sgf

class GoBoardEncoder:
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.num_planes = 17

    def encode(self, game_state):
        """
        Input: game_state object (must have a .board attribute)
        Output: 19x19x17 Tensor
        """
        tensor = np.zeros((self.num_planes, self.board_size, self.board_size), dtype=np.float32)

        # Access the board (Assuming 0=Empty, 1=Black, 2=White)
        board = game_state.board 

        # PLANES 0-2: Stone Status
        for r in range(self.board_size):
            for c in range(self.board_size):
                color = board[r][c]
                if color == 1:   # Black
                    tensor[0][r][c] = 1
                elif color == 2: # White
                    tensor[1][r][c] = 1
                else:            # Empty
                    tensor[2][r][c] = 1
        
        # PLANES 3-10: Liberties 
        # PLANES 11-16: History & Turn
        # Simple Logic: If it's Black's turn, fill Plane 16 with 1s.
        if game_state.color_to_move == 'b':
             tensor[16][:][:] = 1

        return tensor

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Create a Fake Game State to test
    class FakeGame:
        def __init__(self):
            self.board = np.zeros((19, 19), dtype=int)
            self.board[3][3] = 1 # Black Stone
            self.color_to_move = 'b'
    
    encoder = GoBoardEncoder()
    output = encoder.encode(FakeGame())
    print(f" Tensor Created. Shape: {output.shape}")
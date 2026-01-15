import torch
import numpy as np
from captum.attr import IntegratedGradients, NoiseTunnel
from src.model import GoResNet
from src.encoder import GoBoardEncoder

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_heatmap(model, game_state, player_color, target_move_coords):
    """
    Uses Integrated Gradients to calculate pixel attribution (feature importance).
    
    Args:
        model: The trained PyTorch model (GoResNet).
        game_state: SimpleGameState object containing the board and turn info.
        player_color: 'Black' or 'White' - determines perspective.
        target_move_coords: (row, col) tuple of the move we want to explain.
        
    Returns:
        heatmap_norm: 19x19 float array (0.0 to 1.0) showing importance.
        significant_stones: List of dicts [{'color': 'Black', 'coords': (r,c), 'importance': 0.8}, ...]
    """
    # Ensure model is in eval mode for consistent interpretation
    model.eval()
    
    # 1. Prepare Input
    encoder = GoBoardEncoder()
    # Ensure encoding matches the player color perspective
    game_state.color_to_move = 'b' if player_color == 'Black' else 'w'
    
    input_tensor = encoder.encode(game_state)
    # Add batch dimension: [1, 17, 19, 19]
    input_batch = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    input_batch.requires_grad = True

    # 2. Define Target (Convert 2D coord to 1D index for the flat output layer)
    row, col = target_move_coords
    target_idx = row * 19 + col

    # 3. Run XAI (Integrated Gradients)
    ig = IntegratedGradients(model)
    # NoiseTunnel smooths the heatmap, reducing noise and highlighting robust features
    nt = NoiseTunnel(ig)
    
    # Calculate attributions
    # nt_samples: Number of noisy samples to average (higher = smoother but slower)
    attributions, delta = nt.attribute(
        input_batch,
        target=target_idx,
        nt_type='smoothgrad_sq',
        stdevs=0.15,
        nt_samples=5, 
        return_convergence_delta=True
    )

    # 4. Collapse 17 Planes -> 1 Heatmap
    # We sum the absolute values across all 17 feature planes to see "Total Importance" per board spot
    attr_np = attributions.squeeze().cpu().detach().numpy()
    heatmap_2d = np.sum(np.abs(attr_np), axis=0)

    # 5. Normalize (0 to 1) for visualization
    # This ensures the 'hottest' spot is always 1.0 (Brightest Color)
    heatmap_norm = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-9)

    # 6. Extract "Significant Stones" (Cognitive Chunking)
    # Lowered threshold to 0.15 to capture broader "Shape" influence
    significant_stones = []
    board = game_state.board
    for r in range(19):
        for c in range(19):
            # If the AI pays attention to this stone (> 15% importance)
            if heatmap_norm[r][c] > 0.15 and board[r][c] != 0:
                color = "Black" if board[r][c] == 1 else "White"
                # We store the importance score too, to prioritize "Very Important" stones
                significant_stones.append({
                    'color': color, 
                    'coords': (r, c), 
                    'importance': heatmap_norm[r][c]
                })

    # Sort stones by importance (most important first)
    significant_stones.sort(key=lambda x: x['importance'], reverse=True)

    return heatmap_norm, significant_stones
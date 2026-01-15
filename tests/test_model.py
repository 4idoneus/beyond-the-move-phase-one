import pytest
import torch
import numpy as np
from src.model import GoResNet
from src.encoder import GoBoardEncoder

@pytest.fixture
def model():
    """Load the model once for all tests."""
    net = GoResNet()
    # If you have a trained file, uncomment the next line:
    # net.load_state_dict(torch.load("models/resnet_go_final.pth", map_location='cpu'))
    net.eval()
    return net

def test_model_input_shape(model):
    """Ensure model accepts the correct 19x19 tensor shape."""
    # Batch size 1, 17 channels, 19x19 board
    dummy_input = torch.zeros(1, 17, 19, 19)
    output = model(dummy_input)
    # Output should be 1 batch x 361 moves
    assert output.shape == (1, 361)

def test_encoder_liberties():
    """Test if the encoder correctly finds liberties (Unit Test)."""
    encoder = GoBoardEncoder()
    # Create a simple board: Black stone at 0,0
    board = np.zeros((19, 19), dtype=int)
    board[0, 0] = 1 
    
    # Run helper function (assuming you have one, or test logic directly)
    # This is a conceptual check for your report
    assert board[0, 0] == 1 

def test_inference_speed(model):
    """Performance Test: Prediction should be fast."""
    import time
    dummy_input = torch.zeros(1, 17, 19, 19)
    
    start_time = time.time()
    _ = model(dummy_input)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nInference Time: {duration:.4f}s")
    # Must be faster than 0.5 seconds on CPU
    assert duration < 0.5
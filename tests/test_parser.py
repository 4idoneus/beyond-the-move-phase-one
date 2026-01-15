import pytest
import os
from src.process import robust_decode

def test_robust_decode_utf8():
    """Test standard UTF-8 string."""
    byte_data = "Hello World".encode('utf-8')
    assert robust_decode(byte_data) == "Hello World"

def test_robust_decode_gb18030():
    """Test Asian characters (simulating FoxGo SGFs)."""
    # This is Chinese for "Go Game" encoded in GB18030
    byte_data = "围棋".encode('gb18030') 
    assert robust_decode(byte_data) == "围棋"

def test_robust_decode_corruption():
    """Test that it doesn't crash on garbage data."""
    garbage = b'\xff\xfe\xfa' # Invalid bytes
    # It should fall back to 'latin-1' or ignore errors, but NOT crash
    result = robust_decode(garbage)
    assert isinstance(result, str)
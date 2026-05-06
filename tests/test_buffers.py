import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.audio_mock import AudioCapture

def test_ring_buffer_initialization():
    """Buffer should start filled with exact length of zeros."""
    capture = AudioCapture(fs=4000, duration=3.0)
    assert len(capture.buffer) == 12000
    assert np.all(np.array(capture.buffer) == 0.0)

def test_ring_buffer_rollover():
    """Test that appending over length automatically pops old data."""
    capture = AudioCapture(fs=4000, duration=3.0)
    
    # Simulate 1 second of new data (4000 samples of 1.0)
    new_data = np.ones((4000, 1))
    capture._audio_callback(new_data, 4000, None, None)
    
    window = capture.get_latest_window()
    assert len(window) == 12000
    # The last 4000 samples should be 1.0
    assert np.all(window[-4000:] == 1.0)
    # The first 8000 samples should still be 0.0
    assert np.all(window[:8000] == 0.0)

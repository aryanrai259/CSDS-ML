from flask import Flask, render_template, jsonify
import sys
import os
import numpy as np

# Add src to python path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.audio_mock import AudioCapture
from src.dsp.filters import apply_bandpass, normalize_signal
from src.dsp.extractors import generate_mfcc

app = Flask(__name__)

# Global instances (Phase 1 uses laptop mic instead of serial stream)
audio_source = AudioCapture(fs=4000, duration=3.0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stream')
def get_stream_data():
    """
    Returns the latest waveform and extracted features to the frontend.
    This simulates the real-time DSP loop.
    """
    # 1. Pull 3 seconds of data from the ring buffer
    raw_audio = audio_source.get_latest_window()
    
    # 2. DSP Pipeline
    filtered = apply_bandpass(raw_audio)
    normalized = normalize_signal(filtered)
    
    # 3. Feature Extraction
    mfcc = generate_mfcc(normalized)
    
    # 4. Downsample waveform for UI efficiency
    # 12000 points is too heavy for JS Canvas at 30fps. Downsample by 10 -> 1200 points.
    downsampled = normalized[::10].tolist() 
    
    return jsonify({
        "status": "success",
        "waveform": downsampled,
        "mfcc_shape": mfcc.shape
    })

if __name__ == '__main__':
    print("Starting background audio capture...")
    audio_source.start()
    try:
        # Run Flask server
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    finally:
        audio_source.stop()

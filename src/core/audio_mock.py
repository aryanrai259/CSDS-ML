import sounddevice as sd
import numpy as np
import collections

class AudioCapture:
    """
    Captures continuous audio from the laptop microphone in a background thread.
    Uses a thread-safe deque as a ring buffer to store the last N seconds of audio.
    """
    def __init__(self, fs=4000, duration=3.0):
        self.fs = fs
        self.duration = duration
        self.chunk_size = int(fs * duration)
        # Thread-safe ring buffer: automatically discards oldest items when full
        self.buffer = collections.deque(maxlen=self.chunk_size)
        # Pre-fill with zeros to avoid empty buffer errors on boot
        self.buffer.extend(np.zeros(self.chunk_size))
        self.stream = None
        
    def _audio_callback(self, indata, frames, time, status):
        """Called for each audio block by sounddevice."""
        if status:
            print(f"Audio status warning: {status}")
        # indata is shape (frames, channels). Convert to 1D and append to buffer
        self.buffer.extend(indata[:, 0])

    def start(self):
        """Starts the background audio stream."""
        self.stream = sd.InputStream(samplerate=self.fs, channels=1, callback=self._audio_callback)
        self.stream.start()

    def stop(self):
        """Stops the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_latest_window(self):
        """Returns the latest audio chunk as a numpy array."""
        return np.array(self.buffer)

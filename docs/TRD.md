# Technical Requirements Document (TRD)
**Project Name**: Smart Stethoscope: AI-Powered Device for Cardiac and Pulmonary Sound Analysis
**Version**: 2.0 (Deep System Architecture & Data Engineering)

## 1. Comprehensive System Architecture Diagram
This diagram details the exact internal queues, threads, and transformations required.

```mermaid
graph TD
    subgraph Hardware Layer (MCU)
        A[MAX9814 Microphone] -->|Analog Voltage 0-5V| B(Arduino ADC)
        B -->|10-bit integer 0-1023| C{Firmware Main Loop}
        C -->|Raw Bytes via UART| D[USB Interface]
    end

    subgraph Backend Layer (Python Engine)
        D --> E[Serial Thread: PySerial Reader]
        E -->|Push Int16| F[(Thread-safe Ring Buffer)]
        
        subgraph DSP & Inference Pipeline (Worker Thread)
            F -->|Pull 3-sec arrays| G[Data Slicer / Overlap logic]
            G --> H[Bandpass Filter 20Hz-2000Hz]
            H --> I[DC Offset Removal & Normalization]
            I --> J[Librosa MFCC / Spectrogram Generator]
            J -->|Tensor Shape: 1x128x128x1| K[PyTorch/Keras CNN Model]
        end
    end

    subgraph Presentation Layer (Web Server)
        K -->|Prediction Label & Confidence| L[FastAPI / Flask Router]
        F -->|Sub-sampled Waveform| L
        J -->|Base64 Spectrogram Image| L
        L -->|WebSockets / Polling| M[Frontend Dashboard]
    end
```

## 2. Minute Hardware & Firmware Specifications
### 2.1. Arduino ADC Configuration
- **Resolution**: 10-bit ADC (values 0 to 1023).
- **Reference Voltage**: 5V or 3.3V (Must be hardcoded in firmware using `analogReference(EXTERNAL)` if relying on MAX9814 VCC).
- **Timer Interrupts**: Do NOT use standard `delay()`. To achieve an exact 4000 Hz sampling rate, we must configure Arduino Timer1 to trigger an ADC read every 250 microseconds.
- **Baud Rate Calculation**: 4000 samples/sec * ~4 bytes per string transmission (`1023\n`) = 16,000 bytes/sec. `115200` baud gives ~11,520 bytes/sec (INSUFFICIENT if string formatting is used).
  - **Crucial Engineering Decision**: We must transmit raw binary bytes (2 bytes per sample) rather than ASCII strings to prevent serial bottlenecking. 4000 samples * 2 bytes = 8000 bytes/sec, which fits comfortably within 115200 baud.

### 2.2. Microphone (MAX9814) Specs
- **AGC Setting**: Set to Attack/Release ratio of 1:4000 to prevent sharp respiratory sounds from aggressively lowering the gain for subsequent heart sounds.
- **Gain Pin**: Wire to VDD for 40dB gain, or GND for 50dB gain. Start testing at 40dB.

## 3. Mathematical DSP Pipeline (Intricate Steps)
The transformation of the 1D voltage array to a 2D AI feature map requires strict deterministic steps:
1. **DC Offset Removal**: `signal = signal - mean(signal)`
2. **Pre-emphasis**: Apply a high-pass filter to balance the frequency spectrum (boost high frequencies). `signal[t] = signal[t] - α * signal[t-1]`, where α = 0.97.
3. **Framing**: Split the 3-second window into 25ms frames with a 10ms stride. (e.g., 4000Hz * 0.025s = 100 samples per frame).
4. **Windowing**: Apply a Hamming window to each frame to prevent edge artifacts. `frame = frame * hamming(100)`.
5. **Fast Fourier Transform (FFT)**: Compute the 512-point FFT on each frame to get the frequency spectrum.
6. **Mel Filterbank**: Apply 40 triangular Mel-scale filters to the power spectrum.
7. **Log Transformation**: Take the logarithm of the Mel filterbank energies.
8. **Discrete Cosine Transform (DCT)**: Apply DCT to get the final MFCCs. Keep the first 13-40 coefficients.

## 4. Machine Learning Network Topology
If utilizing a CNN for the generated Mel-Spectrograms (image size e.g. 128x128):
- **Input Layer**: `(128, 128, 1)` Grayscale image.
- **Conv Block 1**: Conv2D (32 filters, 3x3), BatchNorm, ReLU, MaxPooling2D (2x2).
- **Conv Block 2**: Conv2D (64 filters, 3x3), BatchNorm, ReLU, MaxPooling2D (2x2).
- **Conv Block 3**: Conv2D (128 filters, 3x3), BatchNorm, ReLU, MaxPooling2D (2x2).
- **Global Average Pooling**: Reduces dimensionality without strict flattening.
- **Dense Layers**: 128 neurons -> Dropout(0.5) -> Output (Softmax).
- **Loss Function**: Categorical Crossentropy.
- **Optimizer**: Adam with learning rate scheduling.

## 5. Data Flow and Concurrency Strategy
A naive sequential loop (`read serial -> process -> infer -> serve ui`) will fail. The serial buffer will overflow while the CNN is calculating.
**Strict Threading Model Required**:
- **Thread 1 (I/O)**: Infinite loop reading from serial and appending to a thread-safe `collections.deque(maxlen=N)`.
- **Thread 2 (DSP & ML)**: Wakes up every 1 second, creates a copy of the last 3 seconds of data from the deque, runs the DSP pipeline, and executes model inference.
- **Thread 3 (Web Server)**: Flask/FastAPI async event loop handling HTTP requests and WebSocket emissions to the frontend.

## 6. Granular Error Handling
1. **Serial Port Unplugged**: Thread 1 throws `serial.SerialException`. Catch it, emit WebSocket event `{"status": "hardware_fault"}`, attempt reconnect every 2 seconds.
2. **Corrupted Byte Stream**: If a byte is dropped, integer reconstruction fails. Implementation must use a synchronization byte (e.g., `0xFF`) to frame the data.
3. **DSP Math Errors**: If an array is completely silent (all zeros), `np.log()` will throw a `DivideByZero` exception. Must add an epsilon value (`np.log(power + 1e-10)`).

## 7. Versioning & Dependencies
- Hardware: Arduino IDE 2.x
- Host: Python 3.10.x
- Key Libs: `pyserial==3.5`, `numpy==1.24.3`, `scipy==1.10.1`, `librosa==0.10.0`, `Flask==2.3.2`, `tensorflow==2.13.0` or `torch==2.0.1`.

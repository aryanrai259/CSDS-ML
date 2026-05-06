// Get Canvas context
const ctx = document.getElementById('waveformChart').getContext('2d');

// Initialize Chart.js instance for the waveform
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        // 3 seconds of audio at 4000Hz = 12000 points. Downsampled by 10 = 1200 points.
        labels: Array.from({length: 1200}, (_, i) => i), 
        datasets: [{
            label: 'Amplitude',
            data: new Array(1200).fill(0), // Pre-allocate fixed size array
            borderColor: '#00ffcc', // Cyberpunk green/cyan
            borderWidth: 1.5,
            pointRadius: 0, // No dots, just line
            tension: 0.1 // Slight curve
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false, // Turn off animation for performance
        scales: {
            y: { 
                min: -1.0, 
                max: 1.0, 
                grid: { color: '#333' } 
            },
            x: { 
                display: false // Hide X axis labels
            }
        },
        plugins: { 
            legend: { display: false } 
        }
    }
});

// Asynchronous polling function to fetch data from the backend
async function fetchStreamData() {
    try {
        const response = await fetch('/api/stream');
        const data = await response.json();
        
        if (data.status === 'success') {
            // Replace data array entirely. Chart.js with parsing: false handles this efficiently.
            chart.data.datasets[0].data = data.waveform;
            chart.update();
            
            // Update diagnostic text
            const shapeStr = `[${data.mfcc_shape.join(', ')}]`;
            document.getElementById('mfccStatus').innerText = `Live | DSP Pipeline Active | MFCC Output Shape: ${shapeStr}`;
            document.getElementById('mfccStatus').style.color = '#4caf50';
        }
    } catch (error) {
        document.getElementById('mfccStatus').innerText = `Connection fault: ${error.message}`;
        document.getElementById('mfccStatus').style.color = '#f44336';
    } finally {
        // Schedule next frame ONLY after the current frame completes rendering
        // This completely prevents promise accumulation and memory leaks
        setTimeout(fetchStreamData, 200); // 5fps for smoother UI without overloading
    }
}

// Start the real-time render loop
fetchStreamData();

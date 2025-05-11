PulseMap


This Python project captures video from a webcam and performs real-time analysis of brightness fluctuations 
in small regions of the image to estimate local heart rate (BPM) systematically. It uses the Fourier transform to 
extract dominant frequency components from temporal brightness changes, revealing periodic signals caused 
by blood flow under the skin. The system then overlays a heatmap of estimated BPM across the video 
and displays live spectral and time-domain plots. The purpose was to identify the veins in real time (e.g. burn victim or 
similar), but the best one can obtain is identifying human body parts in a diverse background.
More generally locating a normal heartbeat in the video. Basically sniper spotting.

Features:
- Real-time webcam input with region-wise brightness tracking
- Fourier-based BPM estimation across image subregions
- Optional display of frequency spectrum and time trace plots
- Live BPM heatmap overlay with filtering for physiologically valid ranges (60–95 BPM)
- Adjustable crop resolution and analysis area

Usage:
1. Install requirements:
   pip install numpy matplotlib opencv-python scipy

2. Run the script:
   python wb6.py

3. Press 'q' to exit the live video feed.

Notes:
- Works best with steady lighting and a stable camera.
- Designed for educational and experimental purposes.
- Performance depends on CPU and camera quality.
- Requires the target to stand absolutely still for a minute in order to work (necessary, not sufficient)


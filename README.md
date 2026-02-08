                                README
Song Version Comparison - Audio and Emotion Analysis
This project is a web-based application designed to analyze, visualize and compare different versions of a musical track(e.g. Original vs. Piano vs.Duet)
It bridges the gap between Digital Signal Processing (DSP) and Generative AI by using Essentia for audio feature extraction and OpenAI (GPT) for semantic emotional interpretation.

Features
1.Multi-Version Analysis: Compare up to 3 versions of a song simultaneously.
2.Audio Feature Extraction:
Calculates Energy, Valence, Roughness, Brightness, Loudness, and Tempo (BPM).
Uses Essentia algorithms for precise signal processing.
3.Smart Segmentation
Automatic: Uses MFCC-based Novelty Curves to detect structural boundaries (Verse/Chorus).
Custom: Allows users to manually define segments via the UI with conflict detection.
4.Dual Visualization Modules
Module A (Analytical): Interactive Bar Charts and Radar Charts using Chart.js.
Module B (Artistic): Dynamic "breathing" motion graphics driven by audio features using p5.js.
5.AI Integration
Uses OpenAI GPT-4o to interpret numerical audio features.
Generates emotional commentary and suggests representative RGB colors for visualization.

Installation & Setup
Prerequisites：Python 3.8+ and An OpenAI API Key (for AI features)

1.Clone the repository
2.Install Dependencies
Create a virtual environment(Linux) and install the required packages:
# Basic requirements
pip install flask numpy openai

# Install Essentia (Example for many Linux/macOS systems via pip)
pip install essentia
3.Set OpenAI API Key
Mac/Linux:
export OPENAI_API_KEY="sk-your-api-key-here"
Note: If no key is provided, the app will run in Mock Mode, returning fake AI data for demonstration.
4.Run the Application
python app.py
5.Access the Web Interface
http://127.0.0.1:5000/view

Usage Guide
1.Upload Audio
In the "Upload your own audio versions" section, select your .wav files and upload them.
Then click "Analyse uploaded audio".
2.Explore Segments(Module A)
Use the Structure Timeline to see the song structure. Click a block to listen to that specific segment.
Use the Bar Charts to compare Energy/Roughness across versions for specific segments.
3.Visual Experience(Module B)
Scroll down to the p5.js canvas.
Observe how the circles breathe (Tempo/Energy) and how the edges vibrate (Roughness).
4.Ask AI
Click the "Ask AI to interpret" button.
The system will send the audio features to GPT.
Read the emotional commentary and watch the visualization colors update automatically based on the AI's suggestion.

Project Structure
1.app.py: The Flask backend. Handles routing, Essentia audio processing, and OpenAI API calls.
2.main.js: Frontend logic controller. Handles data fetching, UI state, and Chart.js rendering.
3.vis.js: p5.js script for artistic visualization.
4.index.html: The main HTML structure (served via /view).


    

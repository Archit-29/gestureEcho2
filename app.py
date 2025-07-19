from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import json
import csv
import os
import pandas as pd
import pickle
import pyttsx3
import threading
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.8)  # Volume level

# Global variables
current_gesture = "none"
last_spoken_phrase = ""
model = None
label_encoder = None
gesture_map = {}

# File paths
DATA_FILE = 'gesture_data.csv'
MODEL_FILE = 'gesture_model.pkl'
ENCODER_FILE = 'label_encoder.pkl'
MAP_FILE = 'gesture_map.json'

def load_gesture_map():
    """Load gesture mapping from JSON file"""
    global gesture_map
    try:
        with open(MAP_FILE, 'r') as f:
            gesture_map = json.load(f)
    except FileNotFoundError:
        gesture_map = {
            "fist": "Hello, how are you?",
            "open_hand": "Thank you very much",
            "peace": "I need help please"
        }
        save_gesture_map()

def save_gesture_map():
    """Save gesture mapping to JSON file"""
    with open(MAP_FILE, 'w') as f:
        json.dump(gesture_map, f, indent=2)

def load_model():
    """Load trained ML model and label encoder"""
    global model, label_encoder
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODER_FILE, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        model = None
        label_encoder = None

def speak_text(text):
    """Speak text using TTS in a separate thread"""
    def speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    
    thread = threading.Thread(target=speak)
    thread.daemon = True
    thread.start()

def predict_gesture(sensor_data):
    """Predict gesture from sensor data"""
    if model is None or label_encoder is None:
        return "model_not_loaded"
    
    try:
        # Convert sensor data to feature array
        features = [
            sensor_data['thumb'],
            sensor_data['index'],
            sensor_data['middle'],
            sensor_data['ring'],
            sensor_data['pinky']
        ]
        
        # Make prediction
        prediction = model.predict([features])
        gesture = label_encoder.inverse_transform(prediction)[0]
        
        return gesture
    except Exception as e:
        print(f"Prediction error: {e}")
        return "error"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sensor_data', methods=['POST'])
def receive_sensor_data():
    """Receive sensor data from ESP32"""
    global current_gesture, last_spoken_phrase
    
    try:
        data = request.json
        print(f"Received: {data}")
        
        # Predict gesture
        predicted_gesture = predict_gesture(data)
        current_gesture = predicted_gesture
        
        # Get phrase and speak it
        if predicted_gesture in gesture_map:
            phrase = gesture_map[predicted_gesture]
            if phrase != last_spoken_phrase:  # Avoid repeating same phrase
                speak_text(phrase)
                last_spoken_phrase = phrase
        
        return jsonify({
            'status': 'success',
            'gesture': predicted_gesture,
            'phrase': gesture_map.get(predicted_gesture, 'Unknown gesture')
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/collect_data', methods=['POST'])
def collect_data():
    """Collect labeled data for training"""
    try:
        data = request.json
        gesture_label = data.get('gesture_label', 'unknown')
        
        # Create CSV header if file doesn't exist
        if not os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['thumb', 'index', 'middle', 'ring', 'pinky', 'gesture', 'timestamp'])
        
        # Append data to CSV
        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                data['thumb'],
                data['index'],
                data['middle'],
                data['ring'],
                data['pinky'],
                gesture_label,
                datetime.now().isoformat()
            ])
        
        return jsonify({'status': 'success', 'message': 'Data collected'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/current_status')
def current_status():
    """Get current gesture and system status"""
    return jsonify({
        'current_gesture': current_gesture,
        'last_phrase': last_spoken_phrase,
        'model_loaded': model is not None,
        'gesture_map': gesture_map
    })

@app.route('/gesture_map', methods=['GET', 'POST'])
def handle_gesture_map():
    """Get or update gesture mapping"""
    if request.method == 'GET':
        return jsonify(gesture_map)
    
    elif request.method == 'POST':
        try:
            new_map = request.json
            gesture_map.update(new_map)
            save_gesture_map()
            return jsonify({'status': 'success', 'message': 'Gesture map updated'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    """Trigger model training"""
    try:
        import subprocess
        result = subprocess.run(['python', 'train_model.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            load_model()  # Reload the newly trained model
            return jsonify({'status': 'success', 'message': 'Model trained successfully'})
        else:
            return jsonify({'status': 'error', 'message': result.stderr}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/data_stats')
def data_stats():
    """Get statistics about collected data"""
    try:
        if not os.path.exists(DATA_FILE):
            return jsonify({'total_samples': 0, 'gestures': {}})
        
        df = pd.read_csv(DATA_FILE)
        gesture_counts = df['gesture'].value_counts().to_dict()
        
        return jsonify({
            'total_samples': len(df),
            'gestures': gesture_counts
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Initialize
    load_gesture_map()
    load_model()
    
    print("GestureEcho backend starting...")
    print("Available gesture mappings:", gesture_map)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
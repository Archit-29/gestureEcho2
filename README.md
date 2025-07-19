# GestureEcho - Smart Glove Speech Assistant

A complete end-to-end system for gesture-based speech assistance using a smart glove with flex sensors.

## 🎯 System Overview

GestureEcho converts hand gestures captured by flex sensors into spoken phrases, enabling communication for mute or deaf individuals.

### Components:
- **Hardware**: ESP32 + 5 flex sensors (one per finger)
- **Backend**: Python Flask server with ML model
- **Frontend**: Web interface for data collection and management
- **ML Model**: Random Forest Classifier for gesture recognition

## 🚀 Quick Start

### 1. Hardware Setup
- Connect 5 flex sensors to ESP32 analog pins (34, 35, 32, 33, 25)
- Update Wi-Fi credentials in `esp32_flex_glove.ino`
- Upload Arduino code to ESP32

### 2. Software Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the Flask backend
python app.py
```

### 3. Web Interface
- Open browser to `http://localhost:5000`
- Collect training data for different gestures
- Train the ML model
- Customize gesture-to-phrase mappings

## 📊 Data Collection Process

1. **Wear the glove** and make a specific gesture
2. **Select gesture label** in the web interface
3. **Click "Collect Sample"** while holding the gesture
4. **Repeat** for different gestures (recommended: 20+ samples per gesture)
5. **Train the model** once you have sufficient data

## 🧠 Machine Learning

The system uses a **Random Forest Classifier** with:
- **Features**: 5 normalized flex sensor values (0.0 - 1.0)
- **Labels**: Gesture names (fist, open_hand, peace, etc.)
- **Training**: Automatic feature importance analysis
- **Persistence**: Model saved as `gesture_model.pkl`

## 🗣️ Text-to-Speech

- Uses `pyttsx3` for offline TTS
- Customizable speech rate and volume
- Prevents repeated phrases for same gesture

## 🎛️ Gesture Mapping

Edit `gesture_map.json` to customize phrases:
```json
{
  "fist": "Hello, how are you?",
  "open_hand": "Thank you very much",
  "peace": "I need help please"
}
```

## 📁 File Structure

```
GestureEcho/
├── app.py                    # Flask backend server
├── train_model.py           # ML model training script
├── esp32_flex_glove.ino     # Arduino code for ESP32
├── gesture_map.json         # Gesture-to-phrase mappings
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html          # Web interface
├── gesture_data.csv        # Training data (generated)
├── gesture_model.pkl       # Trained model (generated)
└── label_encoder.pkl       # Label encoder (generated)
```

## 🔧 Configuration

### ESP32 Configuration
```cpp
// Update these in esp32_flex_glove.ino
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* serverURL = "http://YOUR_SERVER_IP:5000/sensor_data";
```

### Sensor Calibration
Adjust calibration values based on your flex sensors:
```cpp
const int flexMin[5] = {200, 200, 200, 200, 200}; // Straight position
const int flexMax[5] = {800, 800, 800, 800, 800}; // Fully bent position
```

## 🎯 Features

- **Real-time gesture recognition**
- **Customizable phrase mappings**
- **Web-based data collection**
- **Automatic model training**
- **Live sensor data visualization**
- **Speech synthesis**
- **Training data statistics**

## 🔄 Workflow

1. **Setup**: Configure hardware and software
2. **Calibrate**: Adjust sensor ranges for your hand
3. **Collect**: Gather training data for each gesture
4. **Train**: Build ML model from collected data
5. **Map**: Customize gesture-to-phrase associations
6. **Use**: Real-time gesture recognition and speech

## 📈 Performance Tips

- **Collect 20+ samples** per gesture for better accuracy
- **Hold gestures consistently** during data collection
- **Train in similar lighting** conditions to usage
- **Calibrate sensors** for your hand size
- **Use distinctive gestures** for better classification

## 🛠️ Troubleshooting

### ESP32 Issues:
- Check Wi-Fi credentials and network connectivity
- Verify server IP address in Arduino code
- Monitor serial output for debugging

### Model Issues:
- Ensure sufficient training data (10+ samples minimum)
- Check sensor calibration if accuracy is low
- Retrain model after collecting more data

### TTS Issues:
- Install system TTS dependencies if needed
- Adjust speech rate/volume in `app.py`

## 🔮 Future Enhancements

- **WebSocket** for real-time sensor streaming
- **Mobile app** for gesture management
- **Cloud deployment** for remote access
- **Advanced ML models** (Neural Networks)
- **Multi-user support** with user profiles
- **Gesture confidence scoring**

## 📄 License

Open source - feel free to modify and distribute!

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues for improvements.
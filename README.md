# 🎭 Real-Time Emotion Detection (v1)

This project performs **real-time emotion detection** using a webcam.
Faces are detected with OpenCV and emotions are classified using a trained deep learning model.

## 🚀 Features (v1)
- Real-time face detection
- Emotion classification:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- FPS display
- Emotion stabilization (buffer-based smoothing)

## 🛠 Technologies Used
- Python
- OpenCV
- Keras / TensorFlow
- NumPy

## 📂 Project Structure
emotion-detection/
│
├── models/
│ ├── haarcascade_frontalface_default.xml
│ └── emotion_model.h5 (ignored in git)
│
├── main.py
├── README.md
├── .gitignore


## ▶️ How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install opencv-python tensorflow keras numpy
Add the trained model file to:



models/emotion_model.h5
Run:



python main.py
🧠 Roadmap
v2: Age estimation

v3: Gender detection

v4: Performance optimization

v5: Hair & eye color detection

v6: Face recognition & identity memory

⚠️ Notes
The trained .h5 model is not included in the repository.

You must provide your own trained emotion model.

👤 Author
Göktuğ Öztürkmen
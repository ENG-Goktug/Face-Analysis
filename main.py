import cv2
import numpy as np
import time
from keras.models import load_model
from collections import deque, Counter

# =====================
# MODELS
# =====================

face_cascade = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

emotion_model = load_model("models/emotion_model.h5", compile=False)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# =====================
# STABILIZATION
# =====================

emotion_buffer = deque(maxlen=20)   # longer = more stability
current_emotion = "Neutral"

# =====================
# CAMERA
# =====================

cap = cv2.VideoCapture(0)
prev_time = 0

# =====================
# MAIN LOOP
# =====================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        preds = emotion_model.predict(face, verbose=0)[0]
        emotion_idx = np.argmax(preds)
        confidence = preds[emotion_idx]

        # 🔹 confidence kadar oy ekle
        votes = int(confidence * 10) + 1
        for _ in range(votes):
            emotion_buffer.append(emotion_idx)

        # 🔹 en çok oyu alan duygu
        most_common = Counter(emotion_buffer).most_common(1)[0][0]
        current_emotion = emotion_labels[most_common]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{current_emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # =====================
    # FPS
    # =====================

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
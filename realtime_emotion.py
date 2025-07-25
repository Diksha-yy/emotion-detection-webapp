import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("[INFO] Starting webcam...")

# Load the trained model
model = load_model("emotion_model.h5")

# Emotion classes
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

# 🟩 Fix 1: Add namedWindow
cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = grayscale[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=[0, -1])

        prediction = model.predict(roi_gray)
        emotion_label = emotions[np.argmax(prediction)]

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 🟩 Fix 2: Show frame
    cv2.imshow("Emotion Detection", frame)

    # 🟩 Fix 3: Use longer wait time for proper refresh
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("[INFO] Exiting loop...")
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam stopped.")

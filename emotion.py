import cv2
from deepface import DeepFace
import numpy as np

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture from webcam
cap = cv2.VideoCapture(0)

# Emotion colors for better UI
emotion_colors = {
    'happy': (0, 255, 255),      # yellow
    'sad': (255, 0, 0),          # blue
    'angry': (0, 0, 255),        # red
    'fear': (128, 0, 128),       # purple
    'surprise': (0, 255, 0),     # green
    'neutral': (200, 200, 200),  # grey
    'disgust': (0, 165, 255)     # orange
}

# Motivational messages for each emotion
emotion_messages = {
    'happy': "😊 Enjoy every moment!",
    'sad': "💙 Don’t worry, your time will come.",
    'angry': "🔥 Take a deep breath, stay calm.",
    'fear': "😨 Be brave, you’re stronger than you think!",
    'surprise': "😲 Life is full of surprises!",
    'neutral': "😌 Stay relaxed and balanced.",
    'disgust': "😅 Let go of negativity, stay positive!"
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Add border to the webcam frame
    frame = cv2.copyMakeBorder(frame, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Region of interest
        face_roi = rgb[y:y+h, x:x+w]

        # Analyze with DeepFace (emotion, age, gender)
        result = DeepFace.analyze(face_roi, actions=['emotion', 'age', 'gender'], enforce_detection=False)

        dominant_emotion = result[0]['dominant_emotion']
        age = result[0]['age']
        gender = result[0]['dominant_gender']

        # Get emotion probability (for intensity bar)
        emotion_scores = result[0]['emotion']
        intensity = emotion_scores[dominant_emotion]

        # Get color based on emotion
        color = emotion_colors.get(dominant_emotion, (255, 255, 255))

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Emotion, Age, Gender labels
        cv2.putText(frame, f"Emotion: {dominant_emotion.upper()}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Age: {age}", (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Gender: {gender}", (x, y + h + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw Emotion Intensity Bar
        bar_x, bar_y = x, y + h + 70
        bar_width = int(150 * (intensity / 100))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 150, bar_y + 15), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), color, -1)
        cv2.putText(frame, f"{int(intensity)}%", (bar_x + 155, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Motivational message (below face)
        message = emotion_messages.get(dominant_emotion, "")
        cv2.putText(frame, message, (x, y + h + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show final frame
    cv2.imshow('Emotion, Age & Gender Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

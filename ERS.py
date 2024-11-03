import cv2
from fer import FER
import numpy as np

# Initialize the FER detector once
detector = FER()

def process_frame(frame):
    """Process a single frame and return dominant emotion and hp level."""
    # Decode frame if it’s in byte format (e.g., from a web request)
    if isinstance(frame, bytes):
        npimg = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = detector.detect_emotions(frame)
    dominant_emotion = "neutral"
    hp = 100  # default hp value

    if result:
        emotions = result[0]['emotions']
        dominant_emotion = max(emotions, key=emotions.get)
        if dominant_emotion not in ["neutral", "happy"]:
            hp = max(0, hp - 50)  # Example hp adjustment for negative emotions

    return {"emotion": dominant_emotion, "hp": hp}

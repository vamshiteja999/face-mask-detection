"""Face detection (Haar cascade) + mask classification shared by the image and video scripts."""
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from config import IMAGE_SIZE, MODEL_PATH

GREEN = (0, 200, 0)
RED = (0, 0, 255)


class MaskDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model = load_model(model_path)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                  minSize=(60, 60))

    def classify(self, face_bgr):
        """Return (has_mask, confidence) for one cropped face."""
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, IMAGE_SIZE).astype("float32")
        face = preprocess_input(np.expand_dims(face, axis=0))
        with_mask, without_mask = self.model.predict(face, verbose=0)[0]
        return with_mask > without_mask, float(max(with_mask, without_mask))

    def annotate(self, frame):
        """Draw a green box for 'Mask' or a red box for 'No Mask' with the confidence."""
        for (x, y, w, h) in self.detect_faces(frame):
            has_mask, confidence = self.classify(frame[y:y + h, x:x + w])
            color = GREEN if has_mask else RED
            label = f"{'Mask' if has_mask else 'No Mask'}: {confidence * 100:.1f}%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        return frame

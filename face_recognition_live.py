import json
import time
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
import face_recognition
import cv2
from picamera2 import Picamera2


# Function to initialize Firebase app
def initialize_firebase_app():
    cred = credentials.Certificate(
        "surveillance01-a38c9-firebase-adminsdk-fbsvc-fdc94e32a1.json"
    )  # Replace with your Firebase service account key
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://surveillance01-a38c9-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })


# Function to fetch face embeddings from Firebase
def fetch_face_data():
    ref = db.reference("face_embeddings")
    data = ref.get()
    if data:
        known_face_encodings = [np.array(data[entry]["embedding"]) for entry in data]
        known_face_names = [data[entry]["name"] for entry in data]
        return known_face_encodings, known_face_names
    else:
        print("Error: No face embeddings found in Firebase!")
        return [], []


class FaceDetector:
    """Face detector that wraps face_recognition and Picamera2 feed.

    Methods:
    - detect(rgb_frame): return list of (top,right,bottom,left,name)
    - annotate(frame, detections): draw boxes and labels on BGR frame
    - show_feed(): open camera, run loop, display scaled frames
    """

    def __init__(self, known_face_encodings=None, known_face_names=None, display_scale=0.5, camera_size=(640, 480)):
        self.known_face_encodings = known_face_encodings or []
        self.known_face_names = known_face_names or []
        self.display_scale = display_scale
        self.camera_size = camera_size
        self.picam2 = None

    def detect(self, rgb_frame):
        """Detect faces in an RGB frame and match against known encodings.

        Returns a list of (top, right, bottom, left, name).
        """
        detections = []
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            if len(self.known_face_encodings) > 0:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

            detections.append((top, right, bottom, left, name))

        return detections

    def annotate(self, frame, detections):
        """Draw bounding boxes and names onto a BGR frame."""
        for top, right, bottom, left, name in detections:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, max(top - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def show_feed(self):
        """Start Picamera2 and show live annotated feed. Press 'q' to quit."""
        self.picam2 = Picamera2()
        try:
            self.picam2.preview_configuration.main.size = self.camera_size
            self.picam2.preview_configuration.main.format = "RGB888"
            self.picam2.configure("preview")
        except Exception:
            # Fallback: try configure without specifying preview
            try:
                self.picam2.configure()
            except Exception:
                pass

        self.picam2.start()

        print("Starting real-time face recognition... Press 'q' to exit.")

        prev_time = time.time()
        frame_count = 0

        try:
            while True:
                frame = self.picam2.capture_array()
                # frame from camera is BGR by default in this script
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                detections = self.detect(rgb_frame)
                annotated = self.annotate(frame, detections)

                # FPS calculation
                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - prev_time
                if elapsed_time > 0.5:
                    fps = frame_count / elapsed_time
                    prev_time = current_time
                    frame_count = 0
                else:
                    fps = 0

                cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Resize for display to make the window smaller
                small = cv2.resize(annotated, (0, 0), fx=self.display_scale, fy=self.display_scale)
                cv2.imshow("Face Recognition", small)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            try:
                self.picam2.stop()
            except Exception:
                pass


if __name__ == "__main__":
    initialize_firebase_app()
    known_face_encodings, known_face_names = fetch_face_data()
    detector = FaceDetector(known_face_encodings=known_face_encodings, known_face_names=known_face_names, display_scale=0.5)
    detector.show_feed()
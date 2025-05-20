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
    cred = credentials.Certificate("surveillance01-a38c9-firebase-adminsdk-fbsvc-fdc94e32a1.json")  # Replace with your Firebase service account key
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://surveillance01-a38c9-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase database URL
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

# Function to run real-time face recognition with FPS display
def run_face_recognition(known_face_encodings, known_face_names):
    # Initialize Picamera2
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    print("Starting real-time face recognition... Press 'q' to exit.")

    prev_time = time.time()
    frame_count = 0

    while True:
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # FPS calculation
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time

        if elapsed_time > 0.5:  # Update FPS every second
            fps = frame_count / elapsed_time
            prev_time = current_time
            frame_count = 0
        else:
            fps = 0  # Avoid division errors in first few frames

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

# Main execution block
if __name__ == "__main__":
    initialize_firebase_app()
    known_face_encodings, known_face_names = fetch_face_data()
    run_face_recognition(known_face_encodings, known_face_names)

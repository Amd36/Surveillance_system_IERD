import RPi.GPIO as GPIO
from lock_control import lock_on, lock_off, cleanup
import time
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
import face_recognition
import cv2
from picamera2 import Picamera2

LOCK_PIN = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK_PIN, GPIO.OUT)
lock_off(LOCK_PIN)

# Function to initialize Firebase app
def initialize_firebase_app():
    # cred = credentials.Certificate("ierd-surveillance-system-firebase-adminsdk-fbsvc-84dcb6307e.json")  # IERD service account key
    cred = credentials.Certificate("surveillance01-a38c9-firebase-adminsdk-fbsvc-fdc94e32a1.json")  # Junayed Service Account Key
    firebase_admin.initialize_app(cred, {
        # 'databaseURL': 'https://ierd-surveillance-system-default-rtdb.asia-southeast1.firebasedatabase.app/'  # IERD database URL
        'databaseURL': 'https://surveillance01-a38c9-default-rtdb.asia-southeast1.firebasedatabase.app/'    # Junayed Database URL
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
        
def trigger_lock(name="Unknown"):
    if name != "Unknown":
        lock_on(LOCK_PIN)
        print("Welcome Mr.", name)
        time.sleep(2)
        lock_off()
    else:
        lock_off(LOCK_PIN)
        print("Unknown Person")
          
		
# Function to run real-time face recognition with FPS display
def run_face_recognition(known_face_encodings, known_face_names):
    # Initialize Picamera2
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    print("Starting real-time face recognition... Press 'q' to exit.")

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
            trigger_lock(name)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()
    cleanup()

# Main execution block
if __name__ == "__main__":
    initialize_firebase_app()
    known_face_encodings, known_face_names = fetch_face_data()
    run_face_recognition(known_face_encodings, known_face_names)

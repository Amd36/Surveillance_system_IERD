import cv2
import time
import numpy as np
import face_recognition
import json
import firebase_admin
from firebase_admin import credentials, db
from picamera2 import Picamera2

# Function to initialize Firebase
def initialize_firebase_app():
    # cred = credentials.Certificate("ierd-surveillance-system-firebase-adminsdk-fbsvc-84dcb6307e.json")  # IERD service account key
    cred = credentials.Certificate("surveillance01-a38c9-firebase-adminsdk-fbsvc-fdc94e32a1.json")  # Junayed Service Account Key
    firebase_admin.initialize_app(cred, {
        # 'databaseURL': 'https://ierd-surveillance-system-default-rtdb.asia-southeast1.firebasedatabase.app/'  # IERD database URL
        'databaseURL': 'https://surveillance01-a38c9-default-rtdb.asia-southeast1.firebasedatabase.app/'    # Junayed Database URL
    })

# Function to capture an image with a countdown
def capture_image():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    person_name = input("Enter the person's name: ")

    start_time = time.time()
    while time.time() - start_time < 5:
        frame = picam2.capture_array()
        elapsed_time = int(5 - (time.time() - start_time))
        cv2.putText(frame, f"Capturing in {elapsed_time}s", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame = picam2.capture_array()
    filename = f"{person_name}.jpg"
    #cv2.imwrite(filename, frame)
    #print(f"Image saved as {filename}")

    picam2.stop()
    cv2.destroyAllWindows()

    return filename, frame, person_name

# Function to extract face embeddings
def get_face_embeddings(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)
    
    if face_encodings:
        return face_encodings[0].tolist()
    else:
        print("No face detected!")
        return None

# Function to store embeddings in a local JSON file
def save_embeddings_locally(name, embedding, json_filename="face_embeddings.json"):
    try:
        with open(json_filename, "r") as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.append({"name": name, "embedding": embedding})

    with open(json_filename, "w") as file:
        json.dump(existing_data, file, indent=4)
    
    print(f"Face embedding saved locally in {json_filename}")

# Function to send embeddings to Firebase
def send_to_firebase(name, embedding):
    ref = db.reference("face_embeddings")
    ref.update({name: {"name": name, "embedding": embedding}})
    print(f"Face embedding for {name} uploaded to Firebase.")

# Main execution block
if __name__ == "__main__":
    initialize_firebase_app()
    filename, frame, person_name = capture_image()
    embedding = get_face_embeddings(frame)

    if embedding:
        #save_embeddings_locally(person_name, embedding)
        send_to_firebase(person_name, embedding)

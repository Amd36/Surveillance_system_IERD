# Surveillance_system_IERD
This is a surveillance system project for IERD, BCSIR. The programs are written in python 3.11.2 for raspberry pi 5.

If you want to replicate the work, ensure you have **libcamera** and **venv** package installed in the system. First, create a virtual environment with access to system packages:

    python3.11 -m venv venv --system-site-packages

Then activate the venv:

    source venv/bin/activate

Finally install the dependencies:

    pip install -r requirements.txt

Update the codes to include your Firebase credentials and database url.

That's it. You should be good to run the scripts. Details are provided in the code should you require.

## What this project does

This repository contains a Raspberry Pi–based surveillance prototype that combines:

- Real‑time weapon/object detection using a TensorFlow Lite YOLOv5 model
- Real‑time face recognition powered by the `face_recognition` (dlib) library
- Cloud‑backed identity storage with Firebase Realtime Database

The system is designed for on‑device inference on a Raspberry Pi 5 (ARM64), using the official Raspberry Pi Camera via the modern libcamera/PiCamera2 stack.

## Hardware and OS requirements

- Raspberry Pi 5 (recommended) with 64‑bit Raspberry Pi OS (Bullseye/Bookworm) and libcamera
- Raspberry Pi Camera Module (v2, HQ, or compatible) connected and enabled
- Internet connectivity (required for Firebase operations)
- Optional: HDMI display/monitor or VNC for viewing OpenCV windows

## Project structure

```
camera_feed.py                      # Minimal camera preview using PiCamera2 + OpenCV
face_recognition_live.py            # Live face recognition using embeddings from Firebase
update_known_faces.py               # Capture a face, compute embedding, upload to Firebase
weapon_inference_from_camera.py     # YOLOv5 TFLite weapon/object detection from camera
classes.txt                         # Class labels for the YOLOv5 TFLite model
requirements.txt                    # Python dependencies (targeted for Raspberry Pi)
exported_models/
    ├── best-fp16-yolov5m.tflite      # YOLOv5 Medium (FP16) TFLite
    └── best-fp16-yolov5n.tflite      # YOLOv5 Nano   (FP16) TFLite
```

## Models and classes

- TFLite models are provided under `exported_models/`.
    - `best-fp16-yolov5n.tflite`: smaller, faster, lower accuracy
    - `best-fp16-yolov5m.tflite`: larger, slower, higher accuracy
- Detection classes are defined in `classes.txt` (one per line):
    - pistol, smartphone, knife, wallet, billete, card

## Setup and installation (Raspberry Pi)

Your existing steps above create a Python 3.11 virtual environment and install the Python packages.

Optional system preparation on Raspberry Pi (only if needed):

```
# Optional: ensure camera stack and PiCamera2 are present
sudo apt update
sudo apt install -y libcamera-apps
sudo apt install -y python3-picamera2

# Reboot after enabling the camera in raspi-config if you haven’t already
```

Notes:

- `picamera2` is included in `requirements.txt` but is also available via apt. Prefer apt on Raspberry Pi if you run into wheel/build issues.
- `tflite-runtime` is pinned in `requirements.txt`. If installation fails, install the appropriate prebuilt wheel for your Pi/OS, or use the package from the Raspberry Pi repository when available.

## Configuration

Face recognition and embedding sync rely on Firebase Realtime Database. You need a Firebase service account key JSON file and your database URL.

Where to configure:

- `update_known_faces.py`
    - Service account JSON file path
    - Realtime Database URL
- `face_recognition_live.py`
    - Service account JSON file path
    - Realtime Database URL

By default, the scripts expect a service account file like:

```
surveillance01-a38c9-firebase-adminsdk-fbsvc-fdc94e32a1.json
```

You can either rename your downloaded key to match this filename and place it in the project root, or edit the scripts to point to your actual file name and database URL.

Realtime Database structure expected by this project:

```
face_embeddings/
    <person_name>:
        name: "<person_name>"
        embedding: [<128-d or 128+-d face encoding array>]
```

## Usage

General note: in all OpenCV windows, press `q` to quit.

### 1) Preview the camera

Run `camera_feed.py` to validate camera operation and libcamera/PiCamera2 configuration.

### 2) Enroll a person’s face (upload to Firebase)

Run `update_known_faces.py`:

- Prompts for a name
- Shows a short countdown, captures a frame
- Extracts a face embedding via `face_recognition`
- Uploads the embedding to Firebase under `face_embeddings/<name>`

If no face is detected, it will print a message and not upload.

### 3) Real‑time face recognition

Run `face_recognition_live.py`:

- Pulls all embeddings from Firebase at startup
- Runs live face detection and recognition on camera frames
- Draws bounding boxes and names; shows a running FPS overlay

Tip: The more complete your enrollment images (lighting/pose), the better the recognition.

### 4) Real‑time weapon/object detection

Run `weapon_inference_from_camera.py`:

- Loads the YOLOv5 FP16 TFLite model (by default `best-fp16-yolov5m.tflite`)
- Preprocesses frames to 640×640 and runs inference on‑device
- Applies Non‑Maximum Suppression (NMS) and draws labeled boxes

You can switch to the nano model by changing `model_path` in the script to `best-fp16-yolov5n.tflite` for higher FPS (potentially lower accuracy).

## Performance and accuracy notes

- Input resolution is 640×640; ensure good lighting and camera focus
- Confidence and IoU thresholds in NMS can be tuned for your environment
- Model choice (`yolov5n` vs `yolov5m`) trades accuracy for speed on the Pi


## Acknowledgements

- Raspberry Pi and the PiCamera2/libcamera stack
- YOLO family of models and the broader open‑source community
- `face_recognition` by Adam Geitgey (built on dlib)

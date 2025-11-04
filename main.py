import threading
import time
from typing import Optional
from collections import defaultdict

import cv2
import numpy as np
from picamera2 import Picamera2

# Local imports (files in the same folder)
from face_recognition_live import FaceDetector, initialize_firebase_app, fetch_face_data
from weapon_inference_from_camera import WeaponDetector
import lock_control


def main():
    # 1) Setup / initialization (take some time here)
    print("Initializing system... setting up Firebase and detectors. This may take a few seconds...")

    # Initialize Firebase (may raise on missing creds)
    initialize_firebase_app()

    # Fetch known faces from Firebase
    known_face_encodings, known_face_names = fetch_face_data()
    print(f"Loaded {len(known_face_names)} known faces.")

    # Initialize detectors (heavy work - do before starting camera)
    face_detector = FaceDetector(known_face_encodings=known_face_encodings,
                                 known_face_names=known_face_names,
                                 display_scale=0.5)

    weapon_detector = WeaponDetector(display_scale=0.5)

    print("Detectors initialized. Starting camera and scheduler threads.")

    # Shared frame buffer and synchronization
    latest_frame = {"frame": None}  # mutable container so threads can see updates
    frame_lock = threading.Lock()
    stop_event = threading.Event()

    # Throttling dictionaries to avoid spamming prints
    last_seen_time = defaultdict(lambda: 0.0)  # for faces
    last_weapon_time = 0.0
    # Lock control state
    lock_engaged = False
    lock_last_change_time = 0.0
    # seconds to wait after last danger before unlocking
    lock_unlock_delay = 10.0

    # Worker: face recognition
    def face_worker(poll_interval: float = 0.5, welcome_cooldown: float = 5.0):
        nonlocal latest_frame, stop_event, last_seen_time
        print("Face worker started.")
        while not stop_event.is_set():
            # grab a copy of the latest frame
            with frame_lock:
                frame_copy = None if latest_frame["frame"] is None else latest_frame["frame"].copy()

            if frame_copy is None:
                time.sleep(0.05)
                continue

            # face_detector.detect expects an RGB frame
            rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            try:
                detections = face_detector.detect(rgb)
            except Exception as e:
                # log and continue
                print(f"Face worker error: {e}")
                time.sleep(poll_interval)
                continue

            now = time.time()
            for top, right, bottom, left, name in detections:
                if name and name != "Unknown":
                    if now - last_seen_time[name] > welcome_cooldown:
                        print(f"Welcome, {name}!")
                        last_seen_time[name] = now

            time.sleep(poll_interval)
        print("Face worker stopped.")

    # Worker: weapon detection
    def weapon_worker(poll_interval: float = 0.3, danger_cooldown: float = 2.0):
        nonlocal latest_frame, stop_event, last_weapon_time
        nonlocal lock_engaged, lock_last_change_time
        print("Weapon worker started.")
        while not stop_event.is_set():
            with frame_lock:
                frame_copy = None if latest_frame["frame"] is None else latest_frame["frame"].copy()

            if frame_copy is None:
                time.sleep(0.05)
                continue

            try:
                start = time.time()
                results, ow, oh = weapon_detector.detect(frame_copy)
                infer_time = time.time() - start
            except Exception as e:
                print(f"Weapon worker error: {e}")
                time.sleep(poll_interval)
                continue

            now = time.time()
            danger_detected = False
            if results:
                # Gather class names from results
                detected_names = []
                danger_detected = False
                for box, conf, class_id in results:
                    try:
                        cname = weapon_detector.class_names[class_id]
                    except Exception:
                        cname = f"class_{class_id}"
                    detected_names.append((cname, conf))
                    # consider pistol and knife as danger — case-insensitive
                    if cname.lower() in ("pistol", "knife"):
                        danger_detected = True

                # Only print once per cooldown
                if now - last_weapon_time > danger_cooldown:
                    names_str = ", ".join([f"{n}({c:.2f})" for n, c in detected_names])
                    # print inference timing
                    print(f"Weapon inference time: {infer_time*1000:.1f} ms")
                    if danger_detected:
                        print(f"DANGER: Weapon detected: {names_str}")
                    else:
                        print(f"Detected: {names_str}")
                    last_weapon_time = now

            # Lock control logic: engage immediately on danger, disengage after cooldown
            try:
                if danger_detected:
                    if not lock_engaged:
                        lock_control.lock_on()
                        print("Lock engaged due to danger detection.")
                        lock_engaged = True
                        lock_last_change_time = now
                else:
                    if lock_engaged and (now - lock_last_change_time) > lock_unlock_delay:
                        lock_control.lock_off()
                        print("Lock disengaged (no danger).")
                        lock_engaged = False
                        lock_last_change_time = now
            except Exception as e:
                print(f"Lock control error: {e}")

            time.sleep(poll_interval)
        print("Weapon worker stopped.")

    # Start worker threads
    f_thread = threading.Thread(target=face_worker, name="FaceWorker", daemon=True)
    w_thread = threading.Thread(target=weapon_worker, name="WeaponWorker", daemon=True)
    f_thread.start()
    w_thread.start()

    # Start camera display loop (only raw feed shown)
    picam2 = Picamera2()
    try:
        picam2.preview_configuration.main.size = (640, 480)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.configure("preview")
    except Exception:
        try:
            picam2.configure()
        except Exception:
            pass

    picam2.start()

    print("Camera started. Press 'q' in the display window to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            # display the raw feed (optionally scaled smaller for convenience)
            display_scale = 0.7
            small = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow("Camera Feed", small)

            # update latest_frame for workers
            with frame_lock:
                latest_frame["frame"] = frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit requested! Shutting down...")
                break

    finally:
        # signal threads to stop and cleanup
        stop_event.set()
        f_thread.join(timeout=2.0)
        w_thread.join(timeout=2.0)
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            # Ensure GPIO cleanup / lock cleanup on exit
            lock_control.cleanup()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
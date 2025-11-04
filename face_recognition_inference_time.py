#!/usr/bin/env python3
"""Measure FaceDetector inference time on `test_data/*` and save results.

For each person subdirectory under `test_data`, this script:
 - loads each image
 - measures time for FaceDetector.detect(rgb_frame)
 - saves an annotated copy into `test_results/<person>/`
 - accumulates inference times and recognized names

At the end a YAML file `test_results/inference_times.yaml` is written with the
per-person list of times and mean time.

Usage:
  python3 face_recognition_inference_time.py
"""
from __future__ import annotations

import time
from pathlib import Path
from collections import defaultdict
import argparse
import sys

import cv2
import numpy as np

# Try to import yaml, fallback to json if not present
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    import json as _json
    YAML_AVAILABLE = False

# Local project imports
from face_recognition_live import FaceDetector, initialize_firebase_app, fetch_face_data


def parse_args():
    p = argparse.ArgumentParser(description="Measure FaceDetector inference times for test_data")
    p.add_argument("--test-data", default="test_data", help="Path to test_data directory")
    p.add_argument("--results", default="test_results", help="Path to save results and annotated images")
    p.add_argument("--classes", default="classes.txt", help="Path to classes.txt with classes to SKIP for face recognition")
    return p.parse_args()


def ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def annotate_and_save(detector: FaceDetector, bgr_frame, detections, out_path: Path):
    """Use detector.annotate to draw boxes and save annotated BGR frame to out_path."""
    annotated = detector.annotate(bgr_frame.copy(), detections)
    cv2.imwrite(str(out_path), annotated)


def main():
    args = parse_args()
    test_root = Path(args.test_data)
    results_root = Path(args.results)
    classes_path = Path(args.classes)

    # read classes to SKIP face recognition for
    skip_classes = set()
    try:
        if classes_path.exists():
            with open(classes_path, 'r') as f:
                for L in f:
                    name = L.strip()
                    if name:
                        skip_classes.add(name)
    except Exception as e:
        print(f"Warning: couldn't read classes file {classes_path}: {e}")

    if not test_root.exists():
        print(f"Test data directory does not exist: {test_root}")
        sys.exit(1)

    ensure_dirs(results_root)

    # Initialize firebase and load known faces (same as used in main app)
    try:
        initialize_firebase_app()
    except Exception as e:
        print(f"Warning: initialize_firebase_app() failed: {e} — continuing (detector may have no known faces)")

    try:
        known_face_encodings, known_face_names = fetch_face_data()
    except Exception as e:
        print(f"Warning: fetch_face_data() failed: {e}")
        known_face_encodings, known_face_names = [], []

    detector = FaceDetector(known_face_encodings=known_face_encodings, known_face_names=known_face_names)

    summary = {}

    # Iterate person subdirectories; skip any whose name is listed in classes.txt
    for person_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
        person = person_dir.name
        if person in skip_classes:
            print(f"Skipping person (listed in classes.txt): {person}")
            continue
        print(f"Processing person: {person}")
        times = []
        recognized_counts = defaultdict(int)
        person_out = results_root / person
        ensure_dirs(person_out)

        for img_path in sorted(person_dir.glob("*.*")):
            # load image (BGR)
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  Skipping unreadable file: {img_path}")
                continue

            # convert to RGB for detector
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                img_rgb = img_bgr

            # time detection
            t0 = time.time()
            try:
                detections = detector.detect(img_rgb)
            except Exception as e:
                print(f"  Detection error on {img_path}: {e}")
                detections = []
            elapsed = time.time() - t0
            times.append(elapsed)

            # record recognized names
            for top, right, bottom, left, name in detections:
                recognized_counts[name] += 1

            # annotate (detector.annotate expects BGR)
            try:
                out_file = person_out / img_path.name
                annotate_and_save(detector, img_bgr, detections, out_file)
            except Exception as e:
                print(f"  Failed to save annotated image for {img_path}: {e}")

            print(f"  {img_path.name}: {len(detections)} faces, inference_time={elapsed*1000:.1f} ms")

        mean_time = float(np.mean(times)) if len(times) else 0.0

        summary[person] = {
            "image_count": len(times),
            "inference_times_seconds": [float(t) for t in times],
            "mean_inference_time_seconds": float(mean_time),
            "recognized_counts": dict(recognized_counts),
            "result_directory": str(person_out),
        }

    # Write YAML (or JSON fallback)
    out_yaml = results_root / "inference_times.yaml"
    try:
        if YAML_AVAILABLE:
            with open(out_yaml, "w") as f:
                yaml.safe_dump(summary, f)
        else:
            # write JSON if PyYAML unavailable
            out_json = results_root / "inference_times.json"
            with open(out_json, "w") as f:
                _json.dump(summary, f, indent=2)
            print(f"PyYAML not available; wrote JSON results to {out_json}")
    except Exception as e:
        print(f"Failed to write YAML/JSON results: {e}")
        sys.exit(2)

    print(f"Done. Results saved to {results_root}")


if __name__ == "__main__":
    main()

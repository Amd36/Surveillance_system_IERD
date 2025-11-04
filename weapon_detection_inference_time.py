#!/usr/bin/env python3
"""Measure WeaponDetector inference times across models for labeled test data.

This script:
 - reads allowed class names from `classes.txt`
 - enumerates TFLite models in `exported_models/` (files ending with .tflite)
 - for each model, runs WeaponDetector on images inside `test_data/<class>`
   only if <class> is listed in `classes.txt`
 - records per-image inference times and per-class/mean statistics
 - saves annotated images to `test_results/<model_name>/<class>/` and writes a
   YAML (`inference_times.yaml`) or JSON fallback with timings per model

Usage:
  python3 weapon_detection_inference_time.py

The script requires the project's `weapon_inference_from_camera.py` to be
present and importable. It tries to use PyYAML; falls back to JSON if missing.
"""
from __future__ import annotations

import time
from pathlib import Path
from collections import defaultdict
import argparse
import sys
import glob
import os

import cv2
import numpy as np

# YAML fallback
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    import json as _json
    YAML_AVAILABLE = False

# Local imports
from weapon_inference_from_camera import WeaponDetector


def parse_args():
    p = argparse.ArgumentParser(description="Measure WeaponDetector inference times for test_data models")
    p.add_argument("--test-data", default="test_data", help="Path to test_data directory")
    p.add_argument("--results", default="test_results", help="Path to save results and annotated images")
    p.add_argument("--models", default="exported_models", help="Directory containing .tflite models")
    p.add_argument("--classes", default="classes.txt", help="Path to classes.txt with one class per line")
    return p.parse_args()


def read_classes(path: Path) -> set:
    s = set()
    try:
        with open(path, 'r') as f:
            for line in f:
                name = line.strip()
                if name:
                    s.add(name)
    except Exception as e:
        print(f"Warning: could not read classes file {path}: {e}")
    return s


def ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def annotate_and_save(detector: WeaponDetector, bgr_frame, results, ow, oh, out_path: Path):
    try:
        annotated = detector.annotate(bgr_frame.copy(), results, ow, oh)
        cv2.imwrite(str(out_path), annotated)
    except Exception as e:
        print(f"    Failed to annotate/save {out_path}: {e}")


def main():
    args = parse_args()
    test_root = Path(args.test_data)
    results_root = Path(args.results)
    models_dir = Path(args.models)
    classes_file = Path(args.classes)

    if not test_root.exists():
        print(f"Test data directory does not exist: {test_root}")
        sys.exit(1)

    if not models_dir.exists():
        print(f"Models directory does not exist: {models_dir}")
        sys.exit(1)

    allowed_classes = read_classes(classes_file)
    if not allowed_classes:
        print(f"No classes found in {classes_file}; no inference will be run.")

    # find tflite models
    model_files = sorted([p for p in models_dir.glob("*.tflite")])
    if not model_files:
        print(f"No .tflite models found in {models_dir}")
        sys.exit(1)

    ensure_dirs(results_root)

    overall = {}

    for model_path in model_files:
        model_name = model_path.stem
        print(f"Running model: {model_name} ({model_path})")
        # create a results folder for this model
        model_out = results_root / model_name
        ensure_dirs(model_out)

        # Initialize detector for this model; pass classes file so detector loads class list
        try:
            detector = WeaponDetector(model_path=str(model_path), classes_path=str(classes_file))
        except Exception as e:
            print(f"  Failed to initialize WeaponDetector for {model_path}: {e}")
            continue

        model_summary = {}

        # Iterate over only directories that match allowed_classes
        for person_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
            class_name = person_dir.name
            if class_name not in allowed_classes:
                print(f"  Skipping {class_name} (not in classes.txt)")
                continue

            print(f"  Processing class: {class_name}")
            times = []
            detections_count = defaultdict(int)
            class_out = model_out / class_name
            ensure_dirs(class_out)

            for img_path in sorted(person_dir.glob("*.*")):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    print(f"    Skipping unreadable file: {img_path}")
                    continue

                # run detection (WeaponDetector.detect expects a frame in the same color space used elsewhere; pass BGR)
                t0 = time.time()
                try:
                    results, ow, oh = detector.detect(img_bgr)
                except Exception as e:
                    print(f"    Detection error on {img_path}: {e}")
                    results = []
                    ow = img_bgr.shape[1]
                    oh = img_bgr.shape[0]
                elapsed = time.time() - t0
                times.append(elapsed)

                # count detections per class name
                for box, conf, class_id in results:
                    try:
                        cname = detector.class_names[class_id]
                    except Exception:
                        cname = f"class_{class_id}"
                    detections_count[cname] += 1

                # annotate and save
                out_file = class_out / img_path.name
                annotate_and_save(detector, img_bgr, results, ow, oh, out_file)

                print(f"    {img_path.name}: {len(results)} detections, time={elapsed*1000:.1f} ms")

            mean_time = float(np.mean(times)) if len(times) else 0.0
            model_summary[class_name] = {
                "image_count": len(times),
                "inference_times_seconds": [float(t) for t in times],
                "mean_inference_time_seconds": float(mean_time),
                "detections_count": dict(detections_count),
                "result_directory": str(class_out),
            }

        overall[model_name] = model_summary

        # write per-model YAML/JSON
        out_yaml = model_out / "inference_times.yaml"
        try:
            if YAML_AVAILABLE:
                with open(out_yaml, 'w') as f:
                    yaml.safe_dump(model_summary, f)
                print(f"  Wrote YAML results to {out_yaml}")
            else:
                out_json = model_out / "inference_times.json"
                with open(out_json, 'w') as f:
                    _json.dump(model_summary, f, indent=2)
                print(f"  PyYAML not available; wrote JSON to {out_json}")
        except Exception as e:
            print(f"  Failed to write results for model {model_name}: {e}")

    # Optionally, write combined summary
    combined_file = results_root / "all_models_inference_summary.yaml"
    try:
        if YAML_AVAILABLE:
            with open(combined_file, 'w') as f:
                yaml.safe_dump(overall, f)
            print(f"Wrote combined YAML to {combined_file}")
        else:
            combined_json = results_root / "all_models_inference_summary.json"
            with open(combined_json, 'w') as f:
                _json.dump(overall, f, indent=2)
            print(f"PyYAML not available; wrote combined JSON to {combined_json}")
    except Exception as e:
        print(f"Failed to write combined summary: {e}")

    print("Done.")


if __name__ == '__main__':
    main()

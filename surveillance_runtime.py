from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import yaml

import lock_control
from face_recognition_live import (
    FaceDetector,
    fetch_face_data,
    initialize_firebase_app,
)
from weapon_inference_from_camera import WeaponDetector


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "system_config.yaml"
VALID_MODES = {"raw", "face", "weapon", "full"}
MODE_LABELS = {
    "raw": "Raw Feed",
    "face": "Face Recognition",
    "weapon": "Weapon Detection",
    "full": "Full System",
}


def load_system_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


class AccessPolicyController:
    """Fuse face and weapon observations and exclusively control lock GPIO."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._mutex = threading.RLock()
        self._stop_event = threading.Event()
        self._config = config
        self.unlock_duration = float(config.get("unlock_duration", 5.0))
        self.weapon_hold_duration = float(
            config.get("weapon_hold_duration", 10.0)
        )
        self.weapon_scan_max_age = float(config.get("weapon_scan_max_age", 4.0))
        self.face_confirmations = max(1, int(config.get("face_confirmations", 2)))
        self.face_confirmation_window = float(
            config.get("face_confirmation_window", 3.0)
        )
        self.face_rearm_delay = float(config.get("face_rearm_delay", 2.0))
        self.unlock_modes = set(config.get("unlock_modes", ["full"]))

        self.active_mode: Optional[str] = None
        self.policy_state = "locked"
        self.commanded_state = "locked"
        self.reason = "Fail-secure startup"
        self.authorized_person: Optional[str] = None
        self.unlock_until: Optional[float] = None
        self.weapon_hold_until: Optional[float] = None
        self.last_weapon_at: Optional[float] = None
        self.last_weapon_scan_at: Optional[float] = None
        self.last_face_at: Optional[float] = None
        self.face_armed = True
        self._face_observations: dict[str, list[float]] = {}

        lock_control.configure(
            lock_pin=int(config.get("lock_pin", 23)),
            indicator_led_pin=int(config.get("indicator_led_pin", 24)),
            lock_active_low=bool(config.get("lock_active_low", True)),
            indicator_led_on_when_locked=bool(
                config.get("indicator_led_on_when_locked", True)
            ),
        )
        if not self.hardware_available:
            self.policy_state = "fault"
            self.reason = "GPIO lock hardware is unavailable"

        self._watchdog = threading.Thread(
            target=self._watchdog_worker,
            name="AccessPolicyWatchdog",
            daemon=True,
        )
        self._watchdog.start()

    @property
    def hardware_available(self) -> bool:
        return bool(lock_control.status()["available"])

    def set_mode(self, mode: str) -> None:
        with self._mutex:
            self.active_mode = mode
            self._face_observations.clear()
            if self.policy_state == "fault" and self.hardware_available:
                self._command_locked("Recovered in a healthy surveillance mode")
            elif self.policy_state == "locked":
                self.reason = (
                    "Waiting for recognized person"
                    if mode in self.unlock_modes
                    else "This mode does not grant access"
                )

    def observe_faces(self, names: list[str]) -> None:
        now = time.monotonic()
        recognized = list(dict.fromkeys(
            name for name in names if name and name != "Unknown"
        ))

        with self._mutex:
            if recognized:
                self.last_face_at = now
            else:
                self._rearm_face_if_absent(now)
                return

            if (
                not self.hardware_available
                or self.active_mode not in self.unlock_modes
                or not self.face_armed
                or self._weapon_hold_active(now)
            ):
                return

            cutoff = now - self.face_confirmation_window
            for name in recognized:
                observations = self._face_observations.setdefault(name, [])
                observations.append(now)
                observations[:] = [seen for seen in observations if seen >= cutoff]
                self.policy_state = "face_pending"
                self.commanded_state = "locked"
                self.reason = (
                    f"Verifying {name} "
                    f"({len(observations)}/{self.face_confirmations})"
                )
                if len(observations) >= self.face_confirmations:
                    if not self._weapon_scan_is_fresh(now):
                        self.reason = "Waiting for a fresh weapon-safety scan"
                        return
                    if not lock_control.lock_off():
                        self._set_fault("Failed to release the physical lock")
                        return
                    self.policy_state = "unlocked"
                    self.commanded_state = "unlocked"
                    self.reason = f"Access granted to {name}"
                    self.authorized_person = name
                    self.unlock_until = now + self.unlock_duration
                    self.face_armed = False
                    self._face_observations.clear()
                    return

    def observe_weapons(self, class_names: list[str]) -> None:
        now = time.monotonic()
        dangerous = any(
            class_name.lower() in {"pistol", "knife"}
            for class_name in class_names
        )
        with self._mutex:
            self.last_weapon_scan_at = now
            if not dangerous:
                return
            self.last_weapon_at = now
            self.weapon_hold_until = now + self.weapon_hold_duration
            self.unlock_until = None
            self.authorized_person = None
            self.face_armed = False
            self._face_observations.clear()
            if not lock_control.lock_on() and self.hardware_available:
                self._set_fault("Failed to engage the physical lock")
                return
            self.policy_state = "weapon_hold"
            self.commanded_state = "locked"
            self.reason = "Pistol or knife detected"

    def fault(self, reason: str) -> None:
        with self._mutex:
            lock_control.lock_on()
            self._set_fault(reason)

    def _set_fault(self, reason: str) -> None:
        self.policy_state = "fault"
        self.commanded_state = "locked"
        self.reason = reason
        self.unlock_until = None
        self.authorized_person = None
        self._face_observations.clear()

    def _command_locked(self, reason: str) -> None:
        if not lock_control.lock_on() and self.hardware_available:
            self._set_fault("Failed to engage the physical lock")
            return
        self.policy_state = "locked"
        self.commanded_state = "locked"
        self.reason = reason
        self.unlock_until = None
        self.authorized_person = None

    def _weapon_hold_active(self, now: float) -> bool:
        return (
            self.weapon_hold_until is not None
            and now < self.weapon_hold_until
        )

    def _weapon_scan_is_fresh(self, now: float) -> bool:
        return (
            self.last_weapon_scan_at is not None
            and now - self.last_weapon_scan_at <= self.weapon_scan_max_age
        )

    def _rearm_face_if_absent(self, now: float) -> None:
        if (
            not self.face_armed
            and self.last_face_at is not None
            and now - self.last_face_at >= self.face_rearm_delay
        ):
            self.face_armed = True
            self._face_observations.clear()

    def _watchdog_worker(self) -> None:
        while not self._stop_event.wait(0.1):
            now = time.monotonic()
            with self._mutex:
                self._rearm_face_if_absent(now)
                if (
                    self.policy_state == "unlocked"
                    and self.unlock_until is not None
                    and now >= self.unlock_until
                ):
                    self._command_locked("Access window expired")
                elif (
                    self.policy_state == "weapon_hold"
                    and self.weapon_hold_until is not None
                    and now >= self.weapon_hold_until
                ):
                    self.weapon_hold_until = None
                    self._command_locked(
                        "Weapon hold cleared; fresh authorization required"
                    )
                elif (
                    self.policy_state == "face_pending"
                    and self.last_face_at is not None
                    and now - self.last_face_at > self.face_confirmation_window
                ):
                    self.policy_state = "locked"
                    self.commanded_state = "locked"
                    self.reason = "Waiting for recognized person"
                    self._face_observations.clear()

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._mutex:
            hardware_status = lock_control.status()
            commanded_state = (
                hardware_status["state"]
                if hardware_status["available"]
                else self.commanded_state
            )
            remaining = 0.0
            if self.policy_state == "unlocked" and self.unlock_until is not None:
                remaining = max(0.0, self.unlock_until - now)
            elif (
                self.policy_state == "weapon_hold"
                and self.weapon_hold_until is not None
            ):
                remaining = max(0.0, self.weapon_hold_until - now)
            return {
                "state": self.policy_state,
                "commanded": commanded_state,
                "reason": self.reason,
                "authorized_person": self.authorized_person,
                "remaining_seconds": round(remaining, 1),
                "hardware": hardware_status,
            }

    def shutdown(self) -> None:
        self._stop_event.set()
        self._watchdog.join(timeout=1.0)


class SurveillanceRuntime:
    """One camera and the workers required by a single surveillance mode."""

    def __init__(
        self,
        mode: str,
        config: dict[str, Any],
        lock_controller: AccessPolicyController,
        progress: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.config = config
        self.lock_controller = lock_controller
        self.progress = progress or (lambda _state, _message: None)

        self.camera = None
        self.face_detector: Optional[FaceDetector] = None
        self.weapon_detector: Optional[WeaponDetector] = None

        self.stop_event = threading.Event()
        self.failure_event = threading.Event()
        self.ready_event = threading.Event()
        self.error: Optional[str] = None

        self.frame_condition = threading.Condition()
        self.latest_frame = None
        self.frame_sequence = 0

        self.output_condition = threading.Condition()
        self.output_frame: Optional[bytes] = None
        self.output_sequence = 0

        self.results_lock = threading.Lock()
        self.face_results: list[tuple] = []
        self.face_result_at = 0.0
        self.weapon_results: list[tuple] = []
        self.weapon_result_at = 0.0

        self.face_ready = threading.Event()
        self.weapon_ready = threading.Event()
        self.threads: list[threading.Thread] = []

        self.capture_fps = 0.0
        self.face_inference_ms: Optional[float] = None
        self.weapon_inference_ms: Optional[float] = None
        self._fps_frame_count = 0
        self._fps_window_started = time.monotonic()

    def start(self, timeout: float = 90.0) -> None:
        self._initialize_processors()
        self._start_camera_and_workers()

        deadline = time.monotonic() + timeout
        while not self.ready_event.wait(timeout=0.1):
            if self.failure_event.is_set():
                raise RuntimeError(self.error or "Pipeline startup failed")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{MODE_LABELS[self.mode]} did not become ready within {timeout:.0f}s"
                )
        self.lock_controller.set_mode(self.mode)

    def _initialize_processors(self) -> None:
        if (
            self.mode in {"weapon", "full"}
            and not self.lock_controller.hardware_available
        ):
            hardware_error = lock_control.status().get("error") or "unknown GPIO error"
            raise RuntimeError(f"Lock GPIO is unavailable: {hardware_error}")

        if self.mode in {"face", "full"}:
            self.progress("initializing", "Loading Firebase face embeddings")
            initialize_firebase_app()
            encodings, names = fetch_face_data()
            if not encodings:
                raise RuntimeError("No face embeddings were returned by Firebase")
            self.face_detector = FaceDetector(
                known_face_encodings=encodings,
                known_face_names=names,
            )

        if self.mode in {"weapon", "full"}:
            self.progress("initializing", "Loading YOLOv5n weapon detector")
            weapon_config = self.config.get("weapon_detection", {})
            model_path = self._project_path(
                weapon_config.get("model_path", "exported_models/yolov5n.tflite")
            )
            classes_path = self._project_path(
                weapon_config.get("classes_path", "classes.txt")
            )
            self.weapon_detector = WeaponDetector(
                model_path=str(model_path),
                classes_path=str(classes_path),
                input_size=(
                    int(weapon_config.get("input_width", 640)),
                    int(weapon_config.get("input_height", 640)),
                ),
                confidence_threshold=float(
                    weapon_config.get("confidence_threshold", 0.7)
                ),
                iou_threshold=float(weapon_config.get("iou_threshold", 0.4)),
            )

    def _project_path(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    def _start_camera_and_workers(self) -> None:
        from picamera2 import Picamera2

        self.progress("starting", "Starting camera and inference workers")
        camera_config = self.config.get("camera", {})
        size = (
            int(camera_config.get("width", 1280)),
            int(camera_config.get("height", 720)),
        )
        pixel_format = str(camera_config.get("format", "RGB888"))

        self.camera = Picamera2()
        self.camera.configure(
            self.camera.create_preview_configuration(
                main={"size": size, "format": pixel_format}
            )
        )
        self.camera.start()

        if self.face_detector is not None:
            self._start_thread(self._face_worker, "FaceWorker")
        if self.weapon_detector is not None:
            self._start_thread(self._weapon_worker, "WeaponWorker")
        self._start_thread(self._capture_worker, "CameraCapture")

    def _start_thread(self, target: Callable[[], None], name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()

    def _capture_worker(self) -> None:
        camera_config = self.config.get("camera", {})
        jpeg_quality = int(camera_config.get("jpeg_quality", 88))
        face_max_age = float(
            self.config.get("face_recognition", {}).get("result_max_age", 2.0)
        )
        weapon_max_age = float(
            self.config.get("weapon_detection", {}).get("result_max_age", 2.0)
        )

        try:
            while not self.stop_event.is_set():
                frame = self.camera.capture_array()
                self._update_capture_fps()

                with self.frame_condition:
                    self.latest_frame = frame
                    self.frame_sequence += 1
                    self.frame_condition.notify_all()

                annotated = frame.copy()
                now = time.monotonic()
                with self.results_lock:
                    face_results = list(self.face_results)
                    face_result_at = self.face_result_at
                    weapon_results = list(self.weapon_results)
                    weapon_result_at = self.weapon_result_at

                if (
                    self.face_detector is not None
                    and now - face_result_at <= face_max_age
                ):
                    self.face_detector.annotate(annotated, face_results)

                if (
                    self.weapon_detector is not None
                    and now - weapon_result_at <= weapon_max_age
                ):
                    height, width = annotated.shape[:2]
                    self.weapon_detector.annotate(
                        annotated, weapon_results, width, height
                    )

                success, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not success:
                    continue

                with self.output_condition:
                    self.output_frame = encoded.tobytes()
                    self.output_sequence += 1
                    self.output_condition.notify_all()

                if self._processors_ready():
                    self.ready_event.set()
        except Exception as error:
            if not self.stop_event.is_set():
                self._fail(f"Camera capture failed: {error}")

    def _face_worker(self) -> None:
        interval = float(
            self.config.get("face_recognition", {}).get("poll_interval", 0.5)
        )
        last_sequence = 0
        try:
            while not self.stop_event.is_set():
                frame, last_sequence = self._wait_for_frame(last_sequence)
                if frame is None:
                    continue
                started = time.monotonic()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detector.detect(rgb_frame)
                elapsed = time.monotonic() - started
                with self.results_lock:
                    self.face_results = results
                    self.face_result_at = time.monotonic()
                    self.face_inference_ms = elapsed * 1000.0
                self.lock_controller.observe_faces([
                    name
                    for _top, _right, _bottom, _left, name in results
                ])
                self.face_ready.set()
                self.stop_event.wait(interval)
        except Exception as error:
            if not self.stop_event.is_set():
                self._fail(f"Face-recognition worker failed: {error}")

    def _weapon_worker(self) -> None:
        weapon_config = self.config.get("weapon_detection", {})
        interval = float(weapon_config.get("poll_interval", 0.3))
        last_sequence = 0
        try:
            while not self.stop_event.is_set():
                frame, last_sequence = self._wait_for_frame(last_sequence)
                if frame is None:
                    continue
                started = time.monotonic()
                results, _width, _height = self.weapon_detector.detect(frame)
                elapsed = time.monotonic() - started
                detected_classes = [
                    self.weapon_detector.class_names[class_id]
                    for _box, _confidence, class_id in results
                    if 0 <= class_id < len(self.weapon_detector.class_names)
                ]
                self.lock_controller.observe_weapons(detected_classes)
                with self.results_lock:
                    self.weapon_results = results
                    self.weapon_result_at = time.monotonic()
                    self.weapon_inference_ms = elapsed * 1000.0
                self.weapon_ready.set()
                self.stop_event.wait(interval)
        except Exception as error:
            if not self.stop_event.is_set():
                self._fail(f"Weapon-detection worker failed: {error}")

    def _wait_for_frame(self, last_sequence: int):
        with self.frame_condition:
            self.frame_condition.wait_for(
                lambda: self.stop_event.is_set()
                or self.frame_sequence > last_sequence,
                timeout=1.0,
            )
            if self.stop_event.is_set() or self.latest_frame is None:
                return None, last_sequence
            return self.latest_frame.copy(), self.frame_sequence

    def _processors_ready(self) -> bool:
        face_is_ready = self.face_detector is None or self.face_ready.is_set()
        weapon_is_ready = self.weapon_detector is None or self.weapon_ready.is_set()
        return face_is_ready and weapon_is_ready

    def _update_capture_fps(self) -> None:
        now = time.monotonic()
        self._fps_frame_count += 1
        elapsed = now - self._fps_window_started
        if elapsed >= 1.0:
            self.capture_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_window_started = now

    def _fail(self, message: str) -> None:
        if self.error is None:
            self.error = message
        self.failure_event.set()
        self.stop_event.set()
        self.lock_controller.fault(message)
        with self.frame_condition:
            self.frame_condition.notify_all()
        with self.output_condition:
            self.output_condition.notify_all()

    def jpeg_frames(self):
        last_sequence = -1
        while not self.stop_event.is_set():
            with self.output_condition:
                self.output_condition.wait_for(
                    lambda: self.stop_event.is_set()
                    or self.output_sequence != last_sequence,
                    timeout=1.0,
                )
                if self.stop_event.is_set():
                    break
                frame = self.output_frame
                last_sequence = self.output_sequence

            if frame is not None:
                yield frame

    def frames(self):
        for frame in self.jpeg_frames():
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + frame
                + b"\r\n"
            )

    def stop(self, timeout: float = 30.0) -> None:
        self.stop_event.set()
        with self.frame_condition:
            self.frame_condition.notify_all()
        with self.output_condition:
            self.output_condition.notify_all()

        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass

        deadline = time.monotonic() + timeout
        for thread in self.threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)

        alive = [thread.name for thread in self.threads if thread.is_alive()]
        if alive:
            raise RuntimeError(
                "Workers did not stop safely: " + ", ".join(alive)
            )

        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        self.camera = None

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        face_max_age = float(
            self.config.get("face_recognition", {}).get("result_max_age", 2.0)
        )
        weapon_max_age = float(
            self.config.get("weapon_detection", {}).get("result_max_age", 2.0)
        )
        with self.results_lock:
            face_results = (
                list(self.face_results)
                if now - self.face_result_at <= face_max_age
                else []
            )
            weapon_results = (
                list(self.weapon_results)
                if now - self.weapon_result_at <= weapon_max_age
                else []
            )

        faces = list(dict.fromkeys(
            name or "Unknown"
            for _top, _right, _bottom, _left, name in face_results
        ))
        weapons = []
        if self.weapon_detector is not None:
            for _box, confidence, class_id in weapon_results:
                class_name = (
                    self.weapon_detector.class_names[class_id]
                    if 0 <= class_id < len(self.weapon_detector.class_names)
                    else f"class_{class_id}"
                )
                weapons.append({
                    "name": class_name,
                    "confidence": round(float(confidence), 2),
                })

        return {
            "connected": self.ready_event.is_set() and not self.failure_event.is_set(),
            "error": self.error,
            "fps": round(self.capture_fps, 1),
            "detections": {"faces": faces, "weapons": weapons},
            "face_inference_ms": (
                round(self.face_inference_ms, 1)
                if self.face_inference_ms is not None
                else None
            ),
            "weapon_inference_ms": (
                round(self.weapon_inference_ms, 1)
                if self.weapon_inference_ms is not None
                else None
            ),
        }


class ModeController:
    def __init__(self, initial_mode: str = "raw") -> None:
        self._mutex = threading.RLock()
        self._config = load_system_config()
        self.lock_controller = AccessPolicyController(
            self._config.get("lock_control", {})
        )
        self.runtime: Optional[SurveillanceRuntime] = None
        self.active_mode: Optional[str] = None
        self.requested_mode = initial_mode
        self.state = "idle"
        self.message = "Waiting to start"
        self.error: Optional[str] = None
        self.transition_started_at: Optional[float] = None
        self._closed = False
        self._transition_thread: Optional[threading.Thread] = None
        self.request_mode(initial_mode)

    def request_mode(self, mode: str) -> bool:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}")

        with self._mutex:
            if self._closed:
                raise RuntimeError("Mode controller is shutting down")
            if self.state in {"stopping", "initializing", "starting"}:
                raise RuntimeError("A mode transition is already in progress")
            if self.state == "ready" and self.active_mode == mode:
                return False

            self.requested_mode = mode
            self.state = "stopping" if self.runtime is not None else "initializing"
            self.message = (
                "Stopping the active pipeline"
                if self.runtime is not None
                else f"Initializing {MODE_LABELS[mode]}"
            )
            self.error = None
            self.transition_started_at = time.monotonic()
            thread = threading.Thread(
                target=self._transition,
                args=(mode,),
                name="ModeTransition",
                daemon=True,
            )
            self._transition_thread = thread
            thread.start()
            return True

    def _transition(self, mode: str) -> None:
        previous: Optional[SurveillanceRuntime] = None
        runtime: Optional[SurveillanceRuntime] = None
        try:
            self.lock_controller.set_mode("transition")
            with self._mutex:
                previous = self.runtime

            if previous is not None:
                previous.stop()
                with self._mutex:
                    if self.runtime is previous:
                        self.runtime = None
                        self.active_mode = None

            self._update_progress("initializing", f"Initializing {MODE_LABELS[mode]}")
            runtime = SurveillanceRuntime(
                mode,
                self._config,
                self.lock_controller,
                progress=self._update_progress,
            )
            runtime.start()

            with self._mutex:
                if self._closed:
                    runtime.stop()
                    return
                self.runtime = runtime
                self.active_mode = mode
                self.state = "ready"
                self.message = f"{MODE_LABELS[mode]} is ready"
                self.error = None

            threading.Thread(
                target=self._monitor_runtime,
                args=(runtime,),
                name="RuntimeMonitor",
                daemon=True,
            ).start()
        except Exception as error:
            if runtime is not None:
                try:
                    runtime.stop()
                except Exception:
                    pass
            with self._mutex:
                self.lock_controller.fault(str(error))
                self.state = "error"
                self.error = str(error)
                self.message = "Mode failed to start"

    def _update_progress(self, state: str, message: str) -> None:
        with self._mutex:
            self.state = state
            self.message = message

    def _monitor_runtime(self, runtime: SurveillanceRuntime) -> None:
        while not runtime.failure_event.wait(timeout=0.5):
            if runtime.stop_event.is_set():
                return
        with self._mutex:
            if self._closed or self.runtime is not runtime:
                return
            runtime_error = runtime.error or "The active pipeline stopped unexpectedly"
            self.lock_controller.fault(runtime_error)
            self.state = "stopping"
            self.message = "Releasing failed pipeline resources"
            self.transition_started_at = time.monotonic()

        stop_error = None
        try:
            runtime.stop()
        except Exception as error:
            stop_error = str(error)

        with self._mutex:
            if self.runtime is not runtime:
                return
            if stop_error is None:
                self.runtime = None
                self.active_mode = None
            self.state = "error"
            self.error = (
                f"{runtime_error}; cleanup failed: {stop_error}"
                if stop_error
                else runtime_error
            )
            self.message = "Active mode failed"

    def ready_runtime(self) -> Optional[SurveillanceRuntime]:
        with self._mutex:
            if self.state == "ready":
                return self.runtime
            return None

    def status(self) -> dict[str, Any]:
        with self._mutex:
            runtime = self.runtime
            state = self.state
            started_at = self.transition_started_at
            access_status = self.lock_controller.status()
            status = {
                "state": state,
                "message": self.message,
                "error": self.error,
                "active": self.active_mode,
                "requested": self.requested_mode,
                "active_label": (
                    MODE_LABELS[self.active_mode] if self.active_mode else None
                ),
                "requested_label": MODE_LABELS[self.requested_mode],
                "transition_elapsed": (
                    round(time.monotonic() - started_at, 1)
                    if started_at is not None
                    and state in {"stopping", "initializing", "starting"}
                    else 0.0
                ),
                "lock": (
                    access_status["hardware"]["state"]
                    if access_status["hardware"]["available"]
                    else "unavailable"
                ),
                "access": access_status,
            }

        camera_status = runtime.status() if runtime is not None else {
            "connected": False,
            "error": self.error,
            "fps": 0.0,
            "detections": {"faces": [], "weapons": []},
            "face_inference_ms": None,
            "weapon_inference_ms": None,
        }
        status["camera"] = camera_status
        return status

    def shutdown(self) -> None:
        with self._mutex:
            self._closed = True
            runtime = self.runtime
            self.runtime = None
            self.active_mode = None
        if runtime is not None:
            try:
                runtime.stop()
            except Exception:
                pass
        self.lock_controller.shutdown()

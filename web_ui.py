import atexit
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import yaml
from flask import Flask, Response, jsonify, render_template


app = Flask(__name__)
CONFIG_PATH = Path(__file__).parent / "config" / "camera.yaml"


def load_camera_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return config.get("camera", {})


def wifi_connected() -> bool:
    """Return whether Linux reports an active wireless interface or route."""
    interfaces = [
        interface
        for interface in os.listdir("/sys/class/net")
        if interface.startswith(("wl", "wlan"))
    ]
    for interface in interfaces:
        carrier = Path("/sys/class/net", interface, "carrier")
        try:
            if carrier.read_text(encoding="ascii").strip() == "1":
                return True
        except OSError:
            continue

    try:
        with Path("/proc/net/route").open(encoding="ascii") as route_file:
            return any(
                fields[1] == "00000000" and fields[7] != "00000000"
                for line in route_file.readlines()[1:]
                if (fields := line.split()) and len(fields) >= 8
            )
    except OSError:
        return False


class CameraUnavailable:
    error = "Camera not found. Connect a Picamera2 device to start the feed."
    connected = False

    def frames(self):
        return

    def stop(self) -> None:
        return


class CameraStream:
    def __init__(self, config: dict[str, Any]) -> None:
        from picamera2 import Picamera2

        size = (int(config.get("width", 1280)), int(config.get("height", 720)))
        self.camera = Picamera2()
        self.camera.configure(
            self.camera.create_preview_configuration(
                main={"size": size, "format": "RGB888"}
            )
        )
        self.frame: Optional[bytes] = None
        self.condition = threading.Condition()
        self.running = True
        self.error: Optional[str] = None
        self.connected = False
        self.ready = threading.Event()
        self.jpeg_quality = int(config.get("jpeg_quality", 88))
        self.camera.start()
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()

    def _capture_frames(self) -> None:
        while self.running:
            try:
                frame = self.camera.capture_array()
                # Match camera_feed.py: Picamera2's RGB888 frame is encoded directly.
                success, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if success:
                    with self.condition:
                        self.frame = encoded.tobytes()
                        self.connected = True
                        self.ready.set()
                        self.condition.notify_all()
            except Exception as error:
                self.error = f"{CameraUnavailable.error} ({error})"
                self.connected = False
                self.running = False
                try:
                    self.camera.stop()
                except Exception:
                    pass
                with self.condition:
                    self.condition.notify_all()

    def frames(self):
        last_frame = None
        while self.running:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.frame is not None and self.frame != last_frame,
                    timeout=1.0,
                )
                frame = self.frame

            if frame is None:
                continue

            last_frame = frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + frame
                + b"\r\n"
            )

    def stop(self) -> None:
        self.running = False
        with self.condition:
            self.condition.notify_all()
        self.camera.stop()


camera_lock = threading.Lock()


def create_camera_stream() -> CameraStream | CameraUnavailable:
    try:
        stream = CameraStream(load_camera_config())
        if stream.ready.wait(timeout=5.0):
            return stream
        stream.stop()
        unavailable = CameraUnavailable()
        unavailable.error = "Camera not found. No valid frame was received within 5 seconds."
        return unavailable
    except Exception as error:
        unavailable = CameraUnavailable()
        unavailable.error = f"{unavailable.error} ({error})"
        return unavailable


def camera_reconnect_worker() -> None:
    global camera_stream
    while True:
        with camera_lock:
            connected = camera_stream.connected
        if not connected:
            replacement = create_camera_stream()
            # CameraStream becomes connected only after its first valid JPEG.
            if replacement.connected:
                with camera_lock:
                    old_stream = camera_stream
                    camera_stream = replacement
                old_stream.stop()
            elif isinstance(replacement, CameraStream):
                replacement.stop()
        time.sleep(3)


try:
    camera_stream = create_camera_stream()
except Exception:
    camera_stream = CameraUnavailable()
atexit.register(camera_stream.stop)
threading.Thread(target=camera_reconnect_worker, daemon=True).start()


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/video_feed")
def video_feed() -> Response:
    with camera_lock:
        stream = camera_stream
    if not stream.connected:
        return Response(stream.error, status=503, mimetype="text/plain")
    return Response(
        stream.frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status")
def status() -> Response:
    with camera_lock:
        stream = camera_stream
    return jsonify(
        camera={"connected": stream.connected, "error": stream.error},
        wifi={"connected": wifi_connected()},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
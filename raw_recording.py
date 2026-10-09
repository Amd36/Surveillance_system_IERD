"""Raw-mode recordings stay temporary until the operator chooses Save."""
from pathlib import Path
from datetime import datetime
import tempfile
import threading
import time
import shutil
import os
import subprocess
import json
import logging
import queue


class RawRecorder:
    def __init__(self, camera, directory=Path("/home/ierd/recorded_data")):
        self.camera = camera
        self.directory = Path(directory)
        self.mutex = threading.RLock()
        self.encoder = None
        self.output = None
        self.path = None
        self.state = "idle"
        self.message = "Ready to record"
        self.started = 0.0
        self.elapsed = 0.0
        self.failure = threading.Event()
        self.failure_reason = None
        self.commands = queue.Queue()
        self.worker = threading.Thread(target=self._command_worker,
                                       name="RawRecordingWorker", daemon=True)
        self.worker.start()

    def status(self):
        with self.mutex:
            if self.failure.is_set() and self.state == "recording":
                reason = self.failure_reason or "Unknown encoder output error"
                self.discard()
                self.state = "error"
                self.message = "Recording failed: " + reason
            return {"state": self.state, "message": self.message,
                    "elapsed": round(time.monotonic() - self.started, 1)
                    if self.state == "recording" else round(self.elapsed, 1)}

    def _output_failed(self, error):
        # Called by the encoder thread: avoid taking the recorder lock here,
        # since stopping the encoder waits for that thread to finish.
        self.failure_reason = f"{type(error).__name__}: {error}"
        logging.getLogger(__name__).error(
            "Raw recording output failed: %s", self.failure_reason,
            exc_info=(type(error), error, error.__traceback__),
        )
        self.failure.set()

    def start(self):
        with self.mutex:
            if self.state == "recording" or self.path is not None:
                raise RuntimeError("Stop and save or retake the current recording first")
            from picamera2.encoders import H264Encoder
            from picamera2.outputs import FfmpegOutput
            if not shutil.which("ffmpeg"):
                raise RuntimeError("FFmpeg is required for recording")
            self.directory.mkdir(parents=True, exist_ok=True)
            handle, name = tempfile.mkstemp(prefix=".raw-take-", suffix=".mp4", dir=self.directory)
            os.close(handle)
            self.path = Path(name)
            self.failure.clear()
            self.failure_reason = None
            self.encoder = H264Encoder(bitrate=4000000)
            self.output = FfmpegOutput(str(self.path))
            self.output.timeout = 5
            self.output.error_callback = self._output_failed
            try:
                self.camera.start_encoder(self.encoder, self.output, name="main")
            except Exception:
                self.discard()
                raise
            self.started = time.monotonic()
            self.elapsed = 0.0
            self.state = "recording"
            self.message = "Recording"

    def stop(self):
        with self.mutex:
            if self.state != "recording":
                raise RuntimeError("No recording is in progress")
            self.elapsed = time.monotonic() - self.started
            try:
                self.camera.stop_encoder(self.encoder)
                self.encoder = None
                self.output = None
                if self.failure.is_set() or not self.path.exists() or self.path.stat().st_size == 0:
                    raise RuntimeError("Recording failed; no video was saved")
                self._validate_video()
            except Exception:
                self.discard()
                self.state = "error"
                raise
            self.state = "review"
            self.message = "Save this take or retake"

    def _validate_video(self):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "json", str(self.path)],
            capture_output=True, text=True, timeout=10, check=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if not streams or int(streams[0].get("nb_read_packets", 0)) < 1:
            raise RuntimeError("No video frames were recorded; please try again")

    def save(self):
        with self.mutex:
            if self.state != "review" or self.path is None:
                raise RuntimeError("Stop the recording before saving")
            filename = "raw_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".mp4"
            self.path.rename(self.directory / filename)
            self.path = None
            self.state = "idle"
            self.message = "Saved " + filename

    def discard(self):
        with self.mutex:
            try:
                if self.encoder is not None:
                    try:
                        self.camera.stop_encoder(self.encoder)
                    except Exception:
                        pass
                    finally:
                        # Also clean up a partially started encoder.
                        if self.output is not None and self.output.ffmpeg is not None:
                            self.output.stop()
            finally:
                self.encoder = None
                self.output = None
                if self.path is not None:
                    self.path.unlink(missing_ok=True)
                self.path = None
                self.state = "idle"
                self.elapsed = 0.0
                self.message = "Ready to record"

    def _command_worker(self):
        while True:
            command = self.commands.get()
            if command is None:
                return
            action, completed, result = command
            try:
                result["value"] = self._action(action)
            except Exception as error:
                result["error"] = error
            finally:
                completed.set()

    def action(self, action):
        # FfmpegOutput uses PR_SET_PDEATHSIG. Launch it from this persistent
        # thread, never a Flask request thread that exits after responding.
        if not self.worker.is_alive():
            raise RuntimeError("The recording worker has stopped")
        completed = threading.Event()
        result = {}
        self.commands.put((action, completed, result))
        completed.wait()
        if "error" in result:
            raise result["error"]
        return result["value"]

    def close(self):
        if not self.worker.is_alive():
            return
        try:
            self.action("discard")
        finally:
            self.commands.put(None)
            self.worker.join(timeout=10)

    def _action(self, action):
        with self.mutex:
            try:
                if action == "start":
                    self.start()
                elif action == "stop":
                    self.stop()
                elif action == "save":
                    self.save()
                elif action == "discard":
                    self.discard()
                elif action == "retake":
                    if self.state != "review":
                        raise RuntimeError("Stop the recording before retaking")
                    self.discard()
                    self.start()
                else:
                    raise ValueError("Unknown recording action")
            except Exception as error:
                logging.getLogger(__name__).exception("Raw recording action %s failed", action)
                self.message = str(error)
                raise
            return self.status()

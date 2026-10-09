import atexit
import os
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from surveillance_runtime import MODE_LABELS, ModeController


app = Flask(__name__)
mode_controller = ModeController(initial_mode="raw")
atexit.register(mode_controller.shutdown)


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


@app.get("/")
def index() -> str:
    return render_template("index.html", modes=MODE_LABELS)


@app.get("/video_feed")
def video_feed() -> Response:
    runtime = mode_controller.ready_runtime()
    if runtime is None:
        return Response(
            "The selected surveillance mode is not ready.",
            status=503,
            mimetype="text/plain",
        )
    return Response(
        runtime.frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/mode")
def change_mode() -> Response:
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode")
    if mode not in MODE_LABELS:
        return jsonify(error="Select a valid surveillance mode."), 400

    try:
        started = mode_controller.request_mode(mode)
    except RuntimeError as error:
        return jsonify(error=str(error)), 409

    return jsonify(
        accepted=True,
        changed=started,
        mode=mode,
        message=(
            f"Starting {MODE_LABELS[mode]}."
            if started
            else f"{MODE_LABELS[mode]} is already active."
        ),
    ), 202 if started else 200


@app.post("/recording")
def recording_action() -> Response:
    payload = request.get_json(silent=True) or {}
    action = payload.get("action")
    if action not in {"start", "stop", "save", "retake"}:
        return jsonify(error="Select a valid recording action."), 400
    try:
        return jsonify(recording=mode_controller.recording_action(action))
    except Exception as error:
        return jsonify(error=str(error)), 409


@app.get("/status")
def status() -> Response:
    system_status = mode_controller.status()
    system_status["wifi"] = {"connected": wifi_connected()}
    return jsonify(system_status)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)

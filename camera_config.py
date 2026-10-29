"""Shared camera configuration for the runtime and standalone capture scripts."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "system_config.yaml"


def load_system_config():
    """Load settings relative to this module, independent of the working directory."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def create_camera(config=None):
    """Return a configured, stopped Picamera2 with the shared stream orientation."""
    from libcamera import Transform
    from picamera2 import Picamera2

    if config is None:
        config = load_system_config()
    settings = config.get("camera", {})
    hflip = settings.get("hflip", False)
    vflip = settings.get("vflip", False)
    if not isinstance(hflip, bool) or not isinstance(vflip, bool):
        raise ValueError("camera.hflip and camera.vflip must be YAML booleans")
    camera = Picamera2()
    try:
        camera.configure(camera.create_preview_configuration(
            main={
                "size": (int(settings.get("width", 1280)),
                         int(settings.get("height", 720))),
                "format": str(settings.get("format", "RGB888")),
            },
            transform=Transform(hflip=hflip, vflip=vflip),
        ))
    except Exception:
        camera.close()
        raise
    return camera

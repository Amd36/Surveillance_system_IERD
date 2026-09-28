"""Run the reusable full-surveillance pipeline in a desktop OpenCV window."""

import cv2
import numpy as np

from surveillance_runtime import (
    AccessPolicyController,
    SurveillanceRuntime,
    load_system_config,
)


def main() -> None:
    config = load_system_config()
    access_controller = AccessPolicyController(config.get("lock_control", {}))
    runtime = SurveillanceRuntime(
        mode="full",
        config=config,
        lock_controller=access_controller,
        progress=lambda state, message: print(f"[{state}] {message}"),
    )

    print("Initializing Full System. This may take a few seconds.")
    try:
        runtime.start()
        print("Full System ready. Press 'q' to quit.")
        for jpeg in runtime.jpeg_frames():
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            cv2.imshow("Full System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        runtime.stop()
        access_controller.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete. The physical lock state was preserved.")


if __name__ == "__main__":
    main()

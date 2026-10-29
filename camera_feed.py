import cv2
from camera_config import create_camera
import numpy as np

# Load stream dimensions, format, and orientation from system_config.yaml.
picam2 = create_camera()

# Start the camera
picam2.start()

# Create a window using OpenCV
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

while True:
    # Capture a frame from the camera in RGB888 format
    frame = picam2.capture_array()

    # Display the frame in the OpenCV window
    cv2.imshow("Camera Feed", frame)

    # Exit the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
picam2.stop()
cv2.destroyAllWindows()

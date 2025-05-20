import cv2
from picamera2 import Picamera2, Preview
import numpy as np

# Initialize the Picamera2 instance
picam2 = Picamera2()

# Create configuration for capturing in RGB888 format
config = picam2.create_preview_configuration({"format": "RGB888"})

# Apply the configuration
picam2.configure(config)

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

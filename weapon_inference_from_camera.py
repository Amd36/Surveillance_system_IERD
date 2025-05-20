import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# Load the TFLite model and allocate tensors
model_path = 'exported_models/best-fp16-yolov5m.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class names
with open('classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to preprocess the image
def preprocess_image(image):
    original_height, original_width = image.shape[:2]
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0), original_width, original_height

# Function to perform Non-Maximum Suppression (NMS)
def non_max_suppression(detections, iou_threshold=0.4):
    boxes, confidences, class_ids = [], [], []
    
    for detection in detections:
        box = detection[:4]
        confidence = detection[4]
        class_id = np.argmax(detection[5:])
        
        if confidence > 0.5:
            boxes.append(box)
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    indices = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, iou_threshold)).flatten()
    
    return [(boxes[i], confidences[i], class_ids[i]) for i in indices]

# Function to draw predictions on the image
def draw_predictions(image, results, original_width, original_height):
    for box, confidence, class_id in results:
        x_center, y_center, width, height = box
        x_min = int((x_center - width / 2) * original_width)
        y_min = int((y_center - height / 2) * original_height)
        x_max = int((x_center + width / 2) * original_width)
        y_max = int((y_center + height / 2) * original_height)
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255 , 0, 0), 1)

# Function to run inference from PiCamera2
def infer_from_camera():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1080, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    
    while True:
        frame = picam2.capture_array()
        input_data, original_width, original_height = preprocess_image(frame)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data)
        results = non_max_suppression(output_data[0])
        draw_predictions(frame, results, original_width, original_height)
        
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    infer_from_camera()

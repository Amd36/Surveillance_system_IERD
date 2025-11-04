import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

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
def preprocess_image(image_path):
    original_image = cv2.imread(image_path)
    original_height, original_width = original_image.shape[:2]
    image = cv2.resize(original_image, (640, 640))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0), original_image, original_width, original_height

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
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

# Function to run inference
def infer_and_display(image_path):
    input_data, original_image, original_width, original_height = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.array(output_data)
    results = non_max_suppression(output_data[0])
    draw_predictions(original_image, results, original_width, original_height)
    cv2.imshow("Detections", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'sample_data/sample06.jpg'
    infer_and_display(image_path)

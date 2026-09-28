import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2


class WeaponDetector:
    """Weapon detector wrapper around a TFLite YOLO model.

    Methods:
    - detect(frame): run inference and return detections
    - annotate(frame, results, original_w, original_h): draw boxes on frame
    - show_frame(frame, window_name='Detections'): display a smaller window
    - run_camera_loop(): convenience to run live detection from PiCamera2
    """

    def __init__(self,
                 model_path='exported_models/yolov5n.tflite',
                 classes_path='classes.txt',
                 input_size=(640, 640),
                 confidence_threshold=0.7,
                 iou_threshold=0.4,
                 display_scale=0.75):
        self.model_path = model_path
        self.classes_path = classes_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.display_scale = display_scale

        # Load model
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load class names
        with open(self.classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

    def preprocess(self, image):
        original_height, original_width = image.shape[:2]
        input_width, input_height = self.input_size
        scale = min(input_width / original_width, input_height / original_height)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height))
        canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        input_data = np.expand_dims(canvas.astype(np.float32) / 255.0, axis=0)
        transform = (scale, pad_x, pad_y)
        return input_data, original_width, original_height, transform

    def non_max_suppression(self, detections):
        boxes, nms_boxes, confidences, class_ids = [], [], [], []

        for detection in detections:
            box = detection[:4]
            objectness = float(detection[4])
            class_scores = detection[5:]
            class_id = int(np.argmax(class_scores)) if len(class_scores) else 0
            class_score = float(class_scores[class_id]) if len(class_scores) else 1.0
            confidence = objectness * class_score

            if confidence > self.confidence_threshold:
                boxes.append(box)
                x_center, y_center, width, height = box
                nms_boxes.append([
                    float(x_center - width / 2),
                    float(y_center - height / 2),
                    float(width),
                    float(height),
                ])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        if len(boxes) == 0:
            return []

        try:
            indices = np.array(cv2.dnn.NMSBoxes(
                nms_boxes,
                confidences,
                self.confidence_threshold,
                self.iou_threshold,
            )).flatten()
        except Exception:
            # If NMSBoxes fails due to format, return all detections above threshold
            return list(zip(boxes, confidences, class_ids))

        return [(boxes[i], confidences[i], class_ids[i]) for i in indices]

    def detect(self, frame):
        """Run inference on a single frame.

        Returns: results, original_width, original_height
        """
        input_data, original_width, original_height, transform = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_data = np.array(output_data)

        # output_data shape may be (1, N, M) depending on model
        detections = output_data[0] if output_data.ndim == 3 else output_data
        results = self.non_max_suppression(detections)

        scale, pad_x, pad_y = transform
        input_width, input_height = self.input_size
        mapped_results = []
        for box, confidence, class_id in results:
            x_center, y_center, width, height = box
            x_center = (float(x_center) * input_width - pad_x) / scale
            y_center = (float(y_center) * input_height - pad_y) / scale
            width = float(width) * input_width / scale
            height = float(height) * input_height / scale
            mapped_results.append((
                np.array([
                    x_center / original_width,
                    y_center / original_height,
                    width / original_width,
                    height / original_height,
                ]),
                confidence,
                class_id,
            ))
        return mapped_results, original_width, original_height

    def annotate(self, image, results, original_width, original_height):
        for box, confidence, class_id in results:
            x_center, y_center, width, height = box
            x_min = int((x_center - width / 2) * original_width)
            y_min = int((y_center - height / 2) * original_height)
            x_max = int((x_center + width / 2) * original_width)
            y_max = int((y_center + height / 2) * original_height)

            x_min = max(0, min(original_width - 1, x_min))
            y_min = max(0, min(original_height - 1, y_min))
            x_max = max(0, min(original_width - 1, x_max))
            y_max = max(0, min(original_height - 1, y_max))
            if x_max <= x_min or y_max <= y_min:
                continue

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            class_name = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else f"class_{class_id}"
            )
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x_min, max(y_min - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return image

    def show_frame(self, frame, window_name='Detections'):
        # make showing frame smaller for display
        small = cv2.resize(frame, (0, 0), fx=self.display_scale, fy=self.display_scale)
        cv2.imshow(window_name, small)

    def run_camera_loop(self):
        picam2 = Picamera2()
        try:
            picam2.preview_configuration.main.size = (1080, 720)
            picam2.preview_configuration.main.format = "RGB888"
            picam2.configure("preview")
        except Exception:
            # Fallback: try configure without changing preview config
            try:
                picam2.configure()
            except Exception:
                pass

        picam2.start()

        try:
            while True:
                frame = picam2.capture_array()
                results, ow, oh = self.detect(frame)
                annotated = self.annotate(frame, results, ow, oh)
                self.show_frame(annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            picam2.stop()


if __name__ == '__main__':
    detector = WeaponDetector(display_scale=0.5)
    detector.run_camera_loop()

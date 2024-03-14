import numpy as np
import cv2


def initialize_detector(prototxt_path, model_path, min_confidence):
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net, classes, colors, min_confidence


def detect_objects(net, image, classes, colors, min_confidence):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            class_idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{classes[class_idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(image, (startX, startY), (endX, endY), colors[class_idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_idx], 2)
    return image


def run_detection_loop():
    prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
    model_path = 'models/MobileNetSSD_deploy.caffemodel'
    min_confidence = 0.6

    net, classes, colors, min_confidence = initialize_detector(prototxt_path, model_path, min_confidence)
    cap = cv2.VideoCapture(0)  # Use 0 for webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(net, frame, classes, colors, min_confidence)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection_loop()

import numpy as np
import cv2


# Global configurations
CONFIG = {
    'image_path': 'street_with_vehicles.jpg',
    'prototxt_path': 'models/MobileNetSSD_deploy.prototxt',
    'model_path': 'models/MobileNetSSD_deploy.caffemodel',
    'min_confidence': 0.6,
    'classes': ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train", "tvmonitor"]
}


def initialize_network(config):
    np.random.seed(543210)
    colors = np.random.uniform(0, 255, size=(len(config['classes']), 3))
    net = cv2.dnn.readNetFromCaffe(config['prototxt_path'], config['model_path'])
    return net, colors


def detect_objects(net, colors, config, frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > config['min_confidence']:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{config['classes'][idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)


def run_detection_loop():
    config = CONFIG
    net, colors = initialize_network(config)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_objects(net, colors, config, frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection_loop()

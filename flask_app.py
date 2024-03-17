from flask import Flask, jsonify
from flask_cors import CORS
import threading
import cv2

app = Flask(__name__)
CORS(app)
cap = None  # Initialize cap here to be globally accessible


def camera_loop():
    global cap
    cap = cv2.VideoCapture(0)  # Move VideoCapture inside the loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Your existing detection and display code here, adjusted for the global cap

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


@app.route('/start', methods=['GET'])
def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        threading.Thread(target=camera_loop).start()
    return jsonify({"message": "Camera started"})


@app.route('/stop', methods=['GET'])
def stop_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if cap and cap.isOpened():
        cap.release()  # This will cause camera_loop to exit the loop
    return jsonify({"message": "Camera stopped"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)

from flask import Flask, jsonify
import threading
from main import run_detection_loop
import cv2

app = Flask(__name__)
camera_on = False  # Global variable to control camera state


def camera_loop():
    global camera_on
    while True:
        run_detection_loop()


@app.route('/start', methods=['GET'])
def start_camera():
    global camera_on
    if not camera_on:
        camera_on = True
        threading.Thread(target=camera_loop).start()  # Start the camera in a separate thread
    return jsonify({"message": "Camera started"})


@app.route('/stop', methods=['GET'])
def stop_camera():
    global camera_on
    camera_on = False  # This will stop the loop in camera_loop
    return jsonify({"message": "Camera stopped"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)

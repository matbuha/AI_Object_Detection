from flask import Flask, request, jsonify
from main import initialize_network, detect_objects, CONFIG
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)


# Assuming the rest of your script is refactored into functions as previously discussed

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return 'File not uploaded', 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('/tmp', filename)
    file.save(filepath)

    frame = cv2.imread(filepath)
    os.remove(filepath)  # Delete the file after reading to save space

    net, colors = initialize_network(CONFIG)
    detect_objects(net, colors, CONFIG, frame)

    # Here, you'd return the detection results. For simplicity, we're just returning a placeholder response.
    return jsonify({"message": "Detection complete", "detections": []})


if __name__ == '__main__':
    app.run(debug=True, port=5000)

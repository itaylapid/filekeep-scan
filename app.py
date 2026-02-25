from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Server is running"

@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # Force crop 10% מכל צד (בטוח משנה תמונה)
    h, w = image.shape[:2]
    cropped = image[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)]

    _, buffer = cv2.imencode(".jpg", cropped)
    return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

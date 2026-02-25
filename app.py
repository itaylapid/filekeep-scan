from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from io import BytesIO
import os
import math

app = Flask(__name__)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def contour_score(contour, image_shape):
    area = cv2.contourArea(contour)
    if area < 1000:
        return 0

    if not cv2.isContourConvex(contour):
        return 0

    rect = cv2.boundingRect(contour)
    x, y, w, h = rect

    ratio = w / float(h)
    if ratio < 0.3 or ratio > 3.5:
        return 0

    image_area = image_shape[0] * image_shape[1]
    area_score = area / image_area

    cx = x + w / 2
    cy = y + h / 2

    center_x = image_shape[1] / 2
    center_y = image_shape[0] / 2

    dist = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    max_dist = math.sqrt(center_x**2 + center_y**2)
    center_score = 1 - (dist / max_dist)

    total_score = area_score * 0.7 + center_score * 0.3
    return total_score


def detect_candidates(image):
    ratio = image.shape[0] / 800.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 800))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            score = contour_score(approx, resized.shape)
            if score > 0:
                pts = (approx.reshape(4, 2) * ratio).tolist()
                candidates.append((score, pts))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # נחזיר רק 3 מובילים
    return [c[1] for c in candidates[:3]]


@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    candidates = detect_candidates(image)

    return jsonify({
        "candidates": candidates
    })


@app.route("/apply_crop", methods=["POST"])
def apply_crop():
    data = request.json
    image_url = data.get("image_url")
    points = data.get("points")

    if not image_url or not points:
        return jsonify({"error": "Missing data"}), 400

    resp = cv2.imdecode(
        np.frombuffer(
            cv2.imencode('.jpg', cv2.imread(image_url))[1],
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    pts = np.array(points, dtype="float32")
    warped = four_point_transform(resp, pts)

    _, buffer = cv2.imencode(".jpg", warped)
    return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")


@app.route("/")
def home():
    return "Scanner running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

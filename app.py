from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)


def order_points(pts):
    pts = pts.reshape(4, 2)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect

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
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def enhance_document(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    blur = cv2.GaussianBlur(l, (101, 101), 0)
    blur = np.where(blur == 0, 1, blur)
    l_corrected = cv2.divide(l, blur, scale=255)

    l_blended = cv2.addWeighted(l, 0.8, l_corrected, 0.2, 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_contrast = clahe.apply(l_blended)

    lab_final = cv2.merge((l_contrast, a, b))
    result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)

    return result


def scan_document(image):
    original = image.copy()

    ratio = image.shape[0] / 800.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 800))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 75, 200)

    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None or hierarchy is None:
        return enhance_document(original)

    hierarchy = hierarchy[0]
    image_area = resized.shape[0] * resized.shape[1]

    best = None
    best_area = 0

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)

        if area < image_area * 0.2:
            continue

        if area > image_area * 0.95:
            continue

        parent = hierarchy[i][3]
        if parent != -1:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and area > best_area:
            best_area = area
            best = approx

    if best is None:
        return enhance_document(original)

    warped = four_point_transform(original, best.reshape(4, 2) * ratio)
    warped = enhance_document(warped)

    return warped


@app.route("/")
def home():
    return "Scanner running"


@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    result = scan_document(image)

    _, buffer = cv2.imencode(".jpg", result)
    return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

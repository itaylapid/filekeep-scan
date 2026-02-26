from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)


# =========================
# Utilities
# =========================

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
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


# =========================
# Method 1: Edge detection
# =========================

def detect_by_edges(resized):
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    imageArea = resized.shape[0] * resized.shape[1]
    best = None
    bestArea = 0

    for c in contours:
        area = cv2.contourArea(c)

        if area < imageArea * 0.2:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and area > bestArea:
            bestArea = area
            best = approx

    return best, bestArea


# =========================
# Method 2: Region detection
# =========================

def detect_by_region(resized):
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Threshold בהירות (מסמכים לרוב בהירים)
    _, thresh = cv2.threshold(l, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # סוגרים חורים
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    imageArea = resized.shape[0] * resized.shape[1]
    best = None
    bestArea = 0

    for c in contours:
        area = cv2.contourArea(c)

        if area < imageArea * 0.2:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and area > bestArea:
            bestArea = area
            best = approx

    return best, bestArea


# =========================
# Main scan
# =========================

def scan_document(image):
    original = image.copy()

    ratio = image.shape[0] / 800.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 800))

    edge_cnt, edge_area = detect_by_edges(resized)
    region_cnt, region_area = detect_by_region(resized)

    # בוחרים את הגדול יותר
    if edge_area > region_area:
        chosen = edge_cnt
    else:
        chosen = region_cnt

    if chosen is None:
        return original

    warped = four_point_transform(
        original,
        chosen.reshape(4, 2) * ratio
    )

    return warped


# =========================
# Flask
# =========================

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

    scanned = scan_document(image)

    _, buffer = cv2.imencode(".jpg", scanned)
    return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

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


def scan_document(image):
    original = image.copy()

    ratio = image.shape[0] / 800.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 800))

    # 1️⃣ טשטוש חזק מאוד
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (31, 31), 0)

    # 2️⃣ Threshold אוטומטי
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3️⃣ סגירת חורים
    kernel = np.ones((25, 25), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4️⃣ מציאת קונטורים
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return original

    # 5️⃣ לבחור את הגדול ביותר
    largest = max(contours, key=cv2.contourArea)

    # 6️⃣ Convex Hull
    hull = cv2.convexHull(largest)

    # 7️⃣ קירוב ל-4 פינות
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(approx) != 4:
        return original

    warped = four_point_transform(
        original,
        approx.reshape(4, 2) * ratio
    )

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

    scanned = scan_document(image)

    _, buffer = cv2.imencode(".jpg", scanned)
    return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from io import BytesIO
import os

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


def scan_document(image):
    original = image.copy()

    # Resize for stable detection
    ratio = image.shape[0] / 800.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 800))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # ---- Choose largest 4-point contour ----
    screenCnt = None
    max_area = 0

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(c)
            if area > max_area:
                screenCnt = approx
                max_area = area

    if screenCnt is None:
        h, w = original.shape[:2]
        warped = original[int(0.05*h):int(0.95*h), int(0.05*w):int(0.95*w)]
    else:
        warped = four_point_transform(
            original,
            screenCnt.reshape(4, 2) * ratio
        )

    # ---- Illumination correction (balanced) ----
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    blur = cv2.GaussianBlur(l, (101, 101), 0)
    blur = np.where(blur == 0, 1, blur)

    l_corrected = cv2.divide(l, blur, scale=255)
    l_mixed = cv2.addWeighted(l, 0.7, l_corrected, 0.3, 0)

    lab_corrected = cv2.merge((l_mixed, a, b))
    warped = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    # ---- Subtle contrast boost ----
    warped = cv2.convertScaleAbs(warped, alpha=1.04, beta=3)

    # ---- Moderate saturation boost ----
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.convertScaleAbs(s, alpha=1.2, beta=0)
    hsv_enhanced = cv2.merge((h, s, v))
    warped = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

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

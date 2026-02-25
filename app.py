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

    x, y, w, h = cv2.boundingRect(contour)

    # יחס לא קיצוני (לא פס צר)
    ratio = w / float(h)
    if ratio < 0.3 or ratio > 3.5:
        return 0

    image_area = image_shape[0] * image_shape[1]
    area_score = area / image_area

    # ---- ניקוד מרכז ----
    cx = x + w / 2
    cy = y + h / 2

    center_x = image_shape[1] / 2
    center_y = image_shape[0] / 2

    dist = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    max_dist = math.sqrt(center_x**2 + center_y**2)
    center_score = 1 - (dist / max_dist)

    # ---- ניקוד קרבה לשוליים (חדש) ----
    margin_threshold = 0.05  # 5% מהתמונה

    left_dist = x / image_shape[1]
    right_dist = (image_shape[1] - (x + w)) / image_shape[1]
    top_dist = y / image_shape[0]
    bottom_dist = (image_shape[0] - (y + h)) / image_shape[0]

    edge_touch_score = 0

    if left_dist < margin_threshold:
        edge_touch_score += 1
    if right_dist < margin_threshold:
        edge_touch_score += 1
    if top_dist < margin_threshold:
        edge_touch_score += 1
    if bottom_dist < margin_threshold:
        edge_touch_score += 1

    edge_touch_score = edge_touch_score / 4  # נרמול ל־0–1

    # ---- שילוב ניקוד ----
    total_score = (
        area_score * 0.6 +
        center_score * 0.2 +
        edge_touch_score * 0.2
    )

    return total_score


def scan_document(image):
    original = image.copy()

    ratio = image.shape[0] / 800.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 800))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_score = 0
    best_contour = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            score = contour_score(approx, resized.shape)
            if score > best_score:
                best_score = score
                best_contour = approx

    if best_contour is None:
        h, w = original.shape[:2]
        warped = original[int(0.05*h):int(0.95*h), int(0.05*w):int(0.95*w)]
    else:
        warped = four_point_transform(
            original,
            best_contour.reshape(4, 2) * ratio
        )

    # ---- תיקון תאורה עדין ----
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    blur = cv2.GaussianBlur(l, (101, 101), 0)
    blur = np.where(blur == 0, 1, blur)

    l_corrected = cv2.divide(l, blur, scale=255)
    l_mixed = cv2.addWeighted(l, 0.7, l_corrected, 0.3, 0)

    lab_corrected = cv2.merge((l_mixed, a, b))
    warped = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    warped = cv2.convertScaleAbs(warped, alpha=1.04, beta=3)

    # ---- רוויה עדינה ----
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

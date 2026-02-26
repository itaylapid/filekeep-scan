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


def smart_score(contour, resized, edged):
    area = cv2.contourArea(contour)
    if area < 1000:
        return 0

    if not cv2.isContourConvex(contour):
        return 0

    x, y, w, h = cv2.boundingRect(contour)
    ratio = w / float(h)

    # יחס סביר בלבד
    if ratio < 0.4 or ratio > 2.5:
        return 0

    image_area = resized.shape[0] * resized.shape[1]
    area_score = area / image_area  # 0–1

    # ---- צפיפות קצוות פנימיים ----
    mask = np.zeros(resized.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)

    edge_pixels = edged[mask == 255]
    if len(edge_pixels) == 0:
        return 0

    edge_density = np.sum(edge_pixels > 0) / len(edge_pixels)

    # טבלה צפופה = הרבה קצוות פנימיים
    edge_penalty = 1 - min(edge_density * 3, 1)

    # ---- כהות מסגרת ----
    border_mask = np.zeros(resized.shape[:2], dtype="uint8")
    cv2.drawContours(border_mask, [contour], -1, 255, 8)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    border_pixels = gray[border_mask == 255]

    if len(border_pixels) > 0:
        border_darkness = np.mean(border_pixels)
        border_penalty = border_darkness / 255
    else:
        border_penalty = 1

    # ---- ניקוד משולב ----
    total_score = (
        area_score * 0.6 +
        edge_penalty * 0.25 +
        border_penalty * 0.15
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
            score = smart_score(approx, resized, edged)
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

    # ---- Illumination correction ----
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    blur = cv2.GaussianBlur(l, (101, 101), 0)
    blur = np.where(blur == 0, 1, blur)

    l_corrected = cv2.divide(l, blur, scale=255)
    l_mixed = cv2.addWeighted(l, 0.7, l_corrected, 0.3, 0)

    lab_corrected = cv2.merge((l_mixed, a, b))
    warped = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    warped = cv2.convertScaleAbs(warped, alpha=1.04, beta=3)

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

from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)


# =============================
# Utilities
# =============================

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


# =============================
# Hough-based document detection
# =============================

def detect_document_by_lines(resized):

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edged,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=int(resized.shape[1] * 0.4),
        maxLineGap=50
    )

    if lines is None:
        return None

    horizontals = []
    verticals = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        length = np.hypot(x2 - x1, y2 - y1)

        # סינון לפי זווית
        if abs(y2 - y1) < 20:  # אופקי
            horizontals.append((x1, y1, x2, y2, length))
        elif abs(x2 - x1) < 20:  # אנכי
            verticals.append((x1, y1, x2, y2, length))

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    # בוחרים את הקווים הארוכים ביותר
    horizontals = sorted(horizontals, key=lambda x: x[4], reverse=True)[:2]
    verticals = sorted(verticals, key=lambda x: x[4], reverse=True)[:2]

    # מיון למעלה/למטה
    horizontals = sorted(horizontals, key=lambda x: x[1])
    top_line = horizontals[0]
    bottom_line = horizontals[1]

    # מיון לשמאל/ימין
    verticals = sorted(verticals, key=lambda x: x[0])
    left_line = verticals[0]
    right_line = verticals[1]

    # חיתוך קווים
    def intersection(l1, l2):
        x1, y1, x2, y2, _ = l1
        x3, y3, x4, y4, _ = l2

        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1

        A2 = y4 - y3
        B2 = x3 - x4
        C2 = A2 * x3 + B2 * y3

        det = A1 * B2 - A2 * B1
        if det == 0:
            return None

        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return [x, y]

    tl = intersection(top_line, left_line)
    tr = intersection(top_line, right_line)
    bl = intersection(bottom_line, left_line)
    br = intersection(bottom_line, right_line)

    if None in [tl, tr, bl, br]:
        return None

    return np.array([tl, tr, br, bl], dtype="float32")


# =============================
# Main
# =============================

def scan_document(image):
    original = image.copy()

    ratio = image.shape[0] / 800.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 800))

    pts = detect_document_by_lines(resized)

    if pts is None:
        return original

    warped = four_point_transform(
        original,
        pts * ratio
    )

    return warped


# =============================
# Flask
# =============================

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

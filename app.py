def scan_document(image):
    h, w = image.shape[:2]

    # חתוך 10% מכל צד
    crop_top = int(0.10 * h)
    crop_bottom = int(0.90 * h)
    crop_left = int(0.10 * w)
    crop_right = int(0.90 * w)

    cropped = image[crop_top:crop_bottom, crop_left:crop_right]

    return cropped

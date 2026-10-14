import cv2
import numpy as np
from skimage.restoration import estimate_sigma
from ultralytics import YOLO

## Measure the image quality ##
def analyze_quality(gray):

    brightness = float(np.mean(gray))

    contrast = float(np.std(gray))

    noise = float(
        estimate_sigma(
            gray,
            average_sigmas=True
        )
    )

    # Lighting variation
    gray_float = gray.astype(np.float32)
    sigma = max(5.0, min(gray.shape[:2]) / 20.0)
    illumination = cv2.GaussianBlur(
        gray_float, (0, 0), sigmaX=sigma
    )
    lighting = float(np.std(illumination))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "noise": noise,
        "lighting": lighting
    }

## Correct skew in text blocks ##
def correct_skew(image):

    # Grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    # Isolate text
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find foreground
    coord = np.column_stack(
        np.where(binary > 0)
    )

    if len(coord) < 10:
        return image.copy()
    angle = cv2.minAreaRect(coord)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Avoid unnecessary rotation
    if abs(angle) < 0.5:
        return image.copy()

    h, w = gray.shape[:2]

    rotation_matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0
    )

    # Rotate the original crop
    corrected = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return corrected

## Apply filters ##
def filter_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Measure image quality
    quality = analyze_quality(gray)
    processed = gray.copy()
    operations = ["grayscale"]

    # CLAHE for brightness/contrast
    if quality["brightness"] < 120 or quality["contrast"] < 35:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        processed = clahe.apply(processed)
        operations.append("CLAHE")

    # Bilateral filter for denoising
    if quality["noise"] > 12:
        processed = cv2.bilateralFilter(
            processed,
            d=5,
            sigmaColor=35,
            sigmaSpace=35
        )
        operations.append("bilateral_denoising")

    # Adaptive thresholding for shadowed text
    if quality["lighting"] > 25:
        min_dim = min(processed.shape)
        block_size = min(
            31,
            min_dim if min_dim % 2 else min_dim - 1
        )

        if block_size >= 3:
            processed = cv2.adaptiveThreshold(
                processed,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                10
            )
            operations.append("adaptive_threshold")

    return processed

## Process the image ##
def process_image(image, yolo_model, padding=5):
    # Check for missing or invalid image
    if (
        image is None
        or image.size == 0
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        return {}

    h, w = image.shape[:2]
    processed_crops = {}

    # Run YOLO
    model = YOLO(yolo_model)
    results = model(image)

    if not results or results[0].boxes is None:
        return {}

    # Extract detection data
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    names = yolo_model.names

    # Process each detection
    for index, (box, class_id) in enumerate(zip(boxes, class_ids)):

        class_name = names[class_id]

        # Skip picture detections
        if class_name.lower() == "picture":
            continue

        x1, y1, x2, y2 = map(int, box)

        # Add padding within image boundaries
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Skip invalid bounding boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Extract crop
        crop = image[y1:y2, x1:x2]

        # Correct skew
        corrected_crop = correct_skew(crop)

        # Apply image filters
        filtered_crop, operations = filter_image(corrected_crop)

        # Store results
        processed_crops[index] = {
            "class_id": int(class_id),
            "class_name": class_name,
            "coordinates": boxes,
            "processed_crop": filtered_crop,
        }

    return processed_crops

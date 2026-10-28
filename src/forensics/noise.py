import numpy as np
import cv2


def analyze_noise(image):
    """
    Estimate image noise using Laplacian variance.
    Low noise may indicate AI-generated images,
    while higher noise often appears in real camera photos.
    """

    # Convert image to grayscale
    gray_image = np.array(image.convert("L"))

    # Compute Laplacian variance (noise measurement)
    noise_value = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    result = {
        "noise_value": round(float(noise_value), 2),
        "forensic_note": ""
    }

    # Simple forensic interpretation
    if noise_value < 50:
        result["forensic_note"] = "Low natural noise detected (Possible AI-generated image)"
    else:
        result["forensic_note"] = "Natural sensor noise detected"

    return result
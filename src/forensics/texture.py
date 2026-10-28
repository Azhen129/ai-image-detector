import numpy as np
import cv2


def analyze_texture(image):
    """
    Analyze texture consistency using grayscale local detail statistics.
    AI-generated images may show over-smoothed or unnatural texture patterns.
    """

    gray = np.array(image.convert("L"))

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = float(np.var(lap))

    gray_float = gray.astype(np.float32)
    mean = cv2.GaussianBlur(gray_float, (9, 9), 0)
    sq_mean = cv2.GaussianBlur(gray_float ** 2, (9, 9), 0)
    local_variance = sq_mean - (mean ** 2)

    variance_map = cv2.normalize(local_variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    result = {
        "texture_score": round(texture_score, 2),
        "texture_map": variance_map,
        "forensic_note": ""
    }

    if texture_score < 80:
        result["forensic_note"] = "Low texture detail detected (possible synthetic smoothing)"
    elif texture_score > 500:
        result["forensic_note"] = "High texture complexity detected"
    else:
        result["forensic_note"] = "Texture distribution appears natural"

    return result
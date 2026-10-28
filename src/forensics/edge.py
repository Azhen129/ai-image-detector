import numpy as np
import cv2


def analyze_edges(image):
    """
    Analyze edge consistency using Canny edge detection.
    """

    # convert to grayscale
    gray = np.array(image.convert("L"))

    # detect edges
    edges = cv2.Canny(gray, 100, 200)

    # calculate edge density
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size

    edge_density = (edge_pixels / total_pixels) * 100

    result = {
        "edge_score": round(edge_density, 2),
        "edge_image": edges,
        "forensic_note": ""
    }

    if edge_density < 3:
        result["forensic_note"] = "Low edge density detected (possible AI smoothing)"
    elif edge_density > 12:
        result["forensic_note"] = "High edge density detected"
    else:
        result["forensic_note"] = "Edge structure appears natural"

    return result
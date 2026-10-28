import numpy as np
import cv2


def analyze_color(image):
    """
    Analyze color consistency using LAB color space.
    AI-generated images can show unusual color distribution or over-smooth channel balance.
    """

    img_rgb = np.array(image.convert("RGB"))
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(img_lab)

    l_std = float(np.std(l_channel))
    a_std = float(np.std(a_channel))
    b_std = float(np.std(b_channel))

    # Simple color inconsistency score
    color_score = (a_std + b_std) / 2

    # Visualization using color variance map
    blur_a = cv2.GaussianBlur(a_channel, (9, 9), 0)
    blur_b = cv2.GaussianBlur(b_channel, (9, 9), 0)

    diff_a = cv2.absdiff(a_channel, blur_a)
    diff_b = cv2.absdiff(b_channel, blur_b)

    color_map = cv2.addWeighted(diff_a, 0.5, diff_b, 0.5, 0)
    color_map = cv2.normalize(color_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    result = {
        "color_score": round(color_score, 2),
        "color_map": color_map,
        "l_std": round(l_std, 2),
        "a_std": round(a_std, 2),
        "b_std": round(b_std, 2),
        "forensic_note": ""
    }

    if color_score < 18:
        result["forensic_note"] = "Low color variance detected (possible synthetic smoothing)"
    elif color_score > 35:
        result["forensic_note"] = "High color variance detected"
    else:
        result["forensic_note"] = "Color distribution appears natural"

    return result
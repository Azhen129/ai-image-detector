from PIL import Image, ImageChops, ImageEnhance
import io
import numpy as np


def analyze_ela(image, quality=90):
    """
    Perform Error Level Analysis (ELA) on an image.
    Returns:
        - ela_image: PIL image for display
        - ela_score: simple intensity score
        - forensic_note: interpretation
    """

    # Save compressed copy in memory
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")

    original = image.convert("RGB")

    # Difference between original and recompressed image
    diff = ImageChops.difference(original, compressed)

    extrema = diff.getextrema()
    max_diff = max(channel_max for _, channel_max in extrema)

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(diff).enhance(scale)

    ela_array = np.array(ela_image)
    ela_score = float(np.mean(ela_array))

    if ela_score > 25:
        forensic_note = "High compression difference detected (Possible manipulation or AI artifacts)"
    else:
        forensic_note = "Low compression difference detected"

    return {
        "ela_image": ela_image,
        "ela_score": round(ela_score, 2),
        "forensic_note": forensic_note
    }
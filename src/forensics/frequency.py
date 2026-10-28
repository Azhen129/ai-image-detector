import numpy as np
import cv2


def analyze_frequency(image):
    """
    Analyze frequency-domain characteristics using FFT.
    AI-generated images may show abnormal high-frequency patterns.
    """

    gray = np.array(image.convert("L"))

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Normalize for display
    display_spectrum = cv2.normalize(
        magnitude_spectrum,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Compute simple forensic score using high-frequency energy
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Central low-frequency square
    low_freq_size = 30
    mask = np.ones_like(magnitude_spectrum, dtype=np.uint8)
    mask[crow - low_freq_size:crow + low_freq_size,
         ccol - low_freq_size:ccol + low_freq_size] = 0

    high_freq_values = magnitude_spectrum[mask == 1]
    high_freq_score = float(np.mean(high_freq_values))

    result = {
        "frequency_score": round(high_freq_score, 2),
        "spectrum_image": display_spectrum,
        "forensic_note": ""
    }

    if high_freq_score > 55:
        result["forensic_note"] = "Unusual high-frequency distribution detected"
    else:
        result["forensic_note"] = "Frequency distribution appears normal"

    return result
def clamp(value, min_value=0.0, max_value=100.0):
    return max(min_value, min(max_value, value))


# -----------------------------
# Normalization helpers
# Higher value = more suspicious of AI
# -----------------------------
def normalize_noise(noise_score):
    if noise_score <= 20:
        return 75
    elif noise_score <= 50:
        return 60
    elif noise_score <= 100:
        return 35
    else:
        return 20


def normalize_frequency(freq_score):
    if freq_score >= 65:
        return 75
    elif freq_score >= 55:
        return 60
    elif freq_score >= 45:
        return 40
    else:
        return 25


def normalize_edges(edge_score):
    if edge_score <= 2:
        return 70
    elif edge_score <= 4:
        return 55
    elif edge_score <= 8:
        return 35
    else:
        return 20


def normalize_color(color_score):
    if color_score <= 15:
        return 70
    elif color_score <= 22:
        return 55
    elif color_score <= 30:
        return 35
    else:
        return 20


def normalize_texture(texture_score):
    if texture_score <= 60:
        return 75
    elif texture_score <= 120:
        return 60
    elif texture_score <= 250:
        return 35
    else:
        return 20


def normalize_metadata(metadata_flag):
    # Missing EXIF is weak evidence only.
    # Many real social media/web images lose metadata.
    return 40 if not metadata_flag else 20


# -----------------------------
# Balanced fusion score
# -----------------------------
def compute_final_score(
    cnn_prob,
    noise_score,
    freq_score,
    edge_score,
    metadata_flag,
    color_score,
    texture_score
):
    """
    Balanced fusion logic.

    Main idea:
    - CNN is important, but it should not overpower the forensic evidence.
    - If CNN says AI but digital forensics look normal, reduce the final score.
    - If CNN says real but digital forensics look suspicious, increase the final score.
    - This helps reduce both false negatives and false positives.
    """

    cnn_percent = cnn_prob * 100.0

    noise_component = normalize_noise(noise_score)
    freq_component = normalize_frequency(freq_score)
    edge_component = normalize_edges(edge_score)
    metadata_component = normalize_metadata(metadata_flag)
    color_component = normalize_color(color_score)
    texture_component = normalize_texture(texture_score)

    forensic_components = [
        noise_component,
        freq_component,
        edge_component,
        metadata_component,
        color_component,
        texture_component
    ]

    forensic_average = sum(forensic_components) / len(forensic_components)
    suspicious_count = sum(1 for value in forensic_components if value >= 55)

    # More balanced weighting:
    # CNN still matters, but digital forensics has stronger influence now.
    if cnn_percent >= 80:
        cnn_weight = 0.60
        forensic_weight = 0.40
    elif cnn_percent <= 20:
        cnn_weight = 0.65
        forensic_weight = 0.35
    elif 40 <= cnn_percent <= 60:
        cnn_weight = 0.50
        forensic_weight = 0.50
    else:
        cnn_weight = 0.55
        forensic_weight = 0.45

    final_score = (cnn_weight * cnn_percent) + (forensic_weight * forensic_average)

    # If CNN is high AI but forensics look mostly normal, reduce false positives.
    if cnn_percent >= 70 and forensic_average < 50:
        final_score -= 14
    elif cnn_percent >= 65 and forensic_average < 55:
        final_score -= 10

    # If CNN is low AI but forensics look suspicious, reduce false negatives.
    if cnn_percent <= 35 and forensic_average >= 60:
        final_score += 10
    elif cnn_percent <= 45 and suspicious_count >= 4:
        final_score += 8

    # Small forensic boost only when several features agree.
    if suspicious_count >= 5:
        final_score += 5
    elif suspicious_count == 4:
        final_score += 3

    return round(clamp(final_score), 2)
import sys
import os
import io
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models import efficientnet_b0

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit, ImageReader
from reportlab.lib.colors import HexColor

from src.forensics.forensics import analyze_metadata
from src.forensics.noise import analyze_noise
from src.forensics.ela import analyze_ela
from src.forensics.frequency import analyze_frequency
from src.forensics.edge import analyze_edges
from src.forensics.color import analyze_color
from src.forensics.texture import analyze_texture
from src.fusion.fusion_score import compute_final_score

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Image Detection System",
    page_icon="🛡️",
    layout="wide"
)


# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }
.stApp { background: radial-gradient(circle at center, #07142a 0%, #040b18 100%); color: #e8eefc; }
.block-container { max-width: 1400px; padding-top: 1.5rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #f4f7ff !important; }

.hero-wrap { text-align: center; padding-top: 1rem; padding-bottom: 1rem; }
.hero-icon { font-size: 3rem; margin-bottom: 0.6rem; }
.hero-title { font-size: 3rem; font-weight: 800; color: #f4f7ff; margin-bottom: 0.7rem; }
.hero-subtitle {
    color: #b7c2d8;
    font-size: 1.02rem;
    line-height: 1.9;
    max-width: 720px;
    margin: 0 auto 2rem auto;
}

.upload-shell {
    max-width: 760px;
    margin: 0 auto;
    text-align: center;
}

.upload-box-label {
    color: #9eabc7;
    font-size: 0.95rem;
    margin-top: 0.75rem;
}

[data-testid="stFileUploader"] {
    background: rgba(18, 26, 47, 0.95);
    border: 1.5px dashed rgba(130, 150, 190, 0.35);
    border-radius: 18px;
    padding: 1.1rem;
    max-width: 760px;
    margin: 0 auto;
    box-shadow: 0 8px 30px rgba(0,0,0,0.18);
}

[data-testid="stFileUploader"] section { padding: 1.4rem 0.7rem; }
[data-testid="stFileUploader"] small { color: #94a4c4 !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploader"] button { border-radius: 12px !important; font-weight: 700 !important; }

.section-label {
    color: #b7c2d8;
    font-size: 0.95rem;
    margin-bottom: 0.7rem;
}

.panel {
    background: rgba(14, 22, 40, 0.78);
    border: 1px solid rgba(118, 135, 173, 0.16);
    border-radius: 18px;
    padding: 1.2rem 1.25rem;
    box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    margin-bottom: 1rem;
}

.result-box {
    border-radius: 16px;
    padding: 1rem 1.1rem;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.result-ai {
    color: #ff8b80;
    background: rgba(120, 45, 45, 0.28);
    border: 1px solid rgba(255, 128, 128, 0.18);
}

.result-real {
    color: #84f4a3;
    background: rgba(32, 98, 58, 0.24);
    border: 1px solid rgba(132, 244, 163, 0.18);
}

.result-warn {
    color: #ffd46e;
    background: rgba(111, 84, 25, 0.24);
    border: 1px solid rgba(255, 212, 110, 0.18);
}

.card-title {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 1rem;
    color: #f4f7ff;
}

.small-label {
    color: #9aa7c0;
    font-size: 0.95rem;
    margin-bottom: 0.35rem;
}

.big-score {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f7fbff;
    margin-bottom: 0.85rem;
}

.summary-text {
    color: #b7c2d8;
    font-size: 0.98rem;
    line-height: 1.7;
}

.summary-title {
    color: #e9efff;
    font-weight: 700;
    margin-top: 0.6rem;
    margin-bottom: 0.4rem;
}

.metric-card {
    background: rgba(14, 22, 40, 0.78);
    border: 1px solid rgba(118, 135, 173, 0.16);
    border-radius: 16px;
    padding: 1rem;
    text-align: left;
}

.metric-label {
    color: #9aa7c0;
    font-size: 0.95rem;
    margin-bottom: 0.35rem;
}

.metric-value {
    color: #f7fbff;
    font-size: 1.45rem;
    font-weight: 800;
}

.stTabs [data-baseweb="tab-list"] { gap: 1rem; }
.stTabs [data-baseweb="tab"] { color: #b7c2d8; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #f4f7ff !important; }

.stButton>button, .stDownloadButton>button {
    border-radius: 12px;
    padding: 0.75rem 1rem;
    font-weight: 700;
}

.loading-box { text-align: center; padding: 5rem 1rem; }
.loading-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f4f7ff;
    margin-bottom: 1rem;
}
.loading-subtitle {
    color: #a8b3c9;
    font-size: 1rem;
    margin-top: 1rem;
}

.note-box {
    background: rgba(18, 26, 47, 0.7);
    border: 1px solid rgba(118, 135, 173, 0.16);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    color: #b7c2d8;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Final model file not found: {model_path}. "
            "Make sure models/final_model.pt exists."
        )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


model = load_model()
classes = ["ai", "real"]


# -----------------------------
# Image preprocessing
# -----------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


def image_to_tensor(pil_img):
    return base_transform(pil_img).unsqueeze(0)


# -----------------------------
# Improved TTA prediction
# -----------------------------
def predict_with_tta(model, image):
    variants = [
        image,
        image.transpose(Image.FLIP_LEFT_RIGHT),
        transforms.functional.adjust_brightness(image, 1.08),
        transforms.functional.adjust_brightness(image, 0.92),
        transforms.functional.adjust_contrast(image, 1.08),
        transforms.functional.adjust_contrast(image, 0.92),
    ]

    probs = []

    with torch.no_grad():
        for img_variant in variants:
            tensor = image_to_tensor(img_variant)
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs.append(probabilities)

    stacked_probs = torch.stack(probs)
    mean_probs = torch.mean(stacked_probs, dim=0)
    std_probs = torch.std(stacked_probs, dim=0)

    return mean_probs, std_probs


# -----------------------------
# Digital forensic score helpers
# -----------------------------
def calculate_digital_forensics_score(
    noise_score,
    freq_score,
    edge_score,
    metadata_flag,
    color_score,
    texture_score
):
    noise_component = 75 if noise_score <= 20 else 60 if noise_score <= 50 else 35 if noise_score <= 100 else 20
    freq_component = 75 if freq_score >= 65 else 60 if freq_score >= 55 else 40 if freq_score >= 45 else 25
    edge_component = 70 if edge_score <= 2 else 55 if edge_score <= 4 else 35 if edge_score <= 8 else 20
    metadata_component = 40 if not metadata_flag else 20
    color_component = 70 if color_score <= 15 else 55 if color_score <= 22 else 35 if color_score <= 30 else 20
    texture_component = 75 if texture_score <= 60 else 60 if texture_score <= 120 else 35 if texture_score <= 250 else 20

    forensic_components = [
        noise_component,
        freq_component,
        edge_component,
        metadata_component,
        color_component,
        texture_component
    ]

    digital_forensics_score = sum(forensic_components) / len(forensic_components)

    return round(digital_forensics_score, 2), {
        "noise_component": noise_component,
        "frequency_component": freq_component,
        "edge_component": edge_component,
        "metadata_component": metadata_component,
        "color_component": color_component,
        "texture_component": texture_component
    }


# -----------------------------
# Grad-CAM helper
# -----------------------------
def generate_gradcam(model, input_tensor, target_class, target_layer):
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    score = output[0, target_class]
    score.backward()

    forward_handle.remove()
    backward_handle.remove()

    grads = gradients[0]
    acts = activations[0]

    weights_cam = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights_cam * acts).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam.cpu().numpy()

    if cam.max() != 0:
        cam = cam / cam.max()

    return cam


def create_heatmap_overlay(pil_image, cam):
    original = np.array(pil_image.convert("RGB"))
    heatmap = cv2.resize(cam, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.7, heatmap_color, 0.3, 0)
    return heatmap_color, overlay


# -----------------------------
# Radar chart
# -----------------------------
def create_radar_chart(cnn_ai, noise_score, freq_score, edge_score, metadata_flag, color_score, texture_score):
    labels = ["CNN", "Noise", "Frequency", "Edge Risk", "Metadata Risk", "Color", "Texture"]

    digital_score, components = calculate_digital_forensics_score(
        noise_score,
        freq_score,
        edge_score,
        metadata_flag,
        color_score,
        texture_score
    )

    values = [
        cnn_ai,
        components["noise_component"],
        components["frequency_component"],
        components["edge_component"],
        components["metadata_component"],
        components["color_component"],
        components["texture_component"]
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_title("Forensic Radar Chart", pad=30)
    plt.tight_layout()

    return fig


# -----------------------------
# PDF helper functions
# -----------------------------
PDF_BG = HexColor("#040b18")
PDF_BG_2 = HexColor("#07142a")
PDF_PANEL = HexColor("#0e1628")
PDF_PANEL_2 = HexColor("#121a2f")
PDF_BORDER = HexColor("#2b3654")
PDF_TEXT = HexColor("#e8eefc")
PDF_MUTED = HexColor("#9aa7c0")
PDF_WHITE = HexColor("#f7fbff")
PDF_AI = HexColor("#ff8b80")
PDF_REAL = HexColor("#84f4a3")
PDF_WARN = HexColor("#ffd46e")


def pdf_color_for_verdict(verdict):
    verdict_lower = str(verdict).lower()
    if "ai" in verdict_lower and "uncertain" not in verdict_lower:
        return PDF_AI
    if "real" in verdict_lower:
        return PDF_REAL
    return PDF_WARN


def pil_or_array_to_reader(image_obj):
    if isinstance(image_obj, Image.Image):
        img = image_obj.convert("RGB")
    elif isinstance(image_obj, np.ndarray):
        arr = image_obj

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if len(arr.shape) == 2:
            img = Image.fromarray(arr).convert("RGB")
        else:
            img = Image.fromarray(arr).convert("RGB")
    else:
        raise ValueError("Unsupported image object for PDF export.")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageReader(buffer), img.size


def figure_to_reader(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    img = Image.open(buffer).convert("RGB")

    out = io.BytesIO()
    img.save(out, format="PNG")
    out.seek(0)

    return ImageReader(out), img.size


def draw_pdf_background(c):
    width, height = A4
    c.setFillColor(PDF_BG)
    c.rect(0, 0, width, height, stroke=0, fill=1)

    c.setFillColor(PDF_BG_2)
    c.circle(width * 0.5, height * 0.96, 260, stroke=0, fill=1)


def draw_footer(c, page_number):
    width, _ = A4
    c.setFillColor(PDF_MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(40, 24, "AI Image Detection System - Generated Report")
    c.drawRightString(width - 40, 24, f"Page {page_number}")


def new_pdf_page(c, page_number):
    c.showPage()
    draw_pdf_background(c)
    draw_footer(c, page_number)


def draw_title(c, title, subtitle=None):
    width, height = A4

    c.setFillColor(PDF_WHITE)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(40, height - 52, title)

    if subtitle:
        c.setFillColor(PDF_MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(40, height - 70, subtitle)


def draw_section_title(c, text, x, y):
    c.setFillColor(PDF_WHITE)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, text)


def draw_card(c, x, y, w, h, label, value, accent_color=PDF_WHITE, value_size=16):
    c.setFillColor(PDF_PANEL)
    c.setStrokeColor(PDF_BORDER)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 10, stroke=1, fill=1)

    c.setFillColor(PDF_MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawString(x + 10, y + h - 18, label)

    c.setFillColor(accent_color)
    c.setFont("Helvetica-Bold", value_size)

    value_text = str(value)
    max_width = w - 18

    if c.stringWidth(value_text, "Helvetica-Bold", value_size) > max_width:
        c.setFont("Helvetica-Bold", 11)

    c.drawString(x + 10, y + 17, value_text)


def draw_wrapped_text(c, text, x, y, max_width, font_name="Helvetica", font_size=9, leading=13, color=PDF_TEXT):
    c.setFillColor(color)
    c.setFont(font_name, font_size)

    lines = simpleSplit(str(text), font_name, font_size, max_width)

    for line in lines:
        c.drawString(x, y, line)
        y -= leading

    return y


def draw_image_fit(c, image_obj, x, y, w, h, caption=None):
    reader, (img_w, img_h) = pil_or_array_to_reader(image_obj)

    scale = min(w / img_w, h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale

    draw_x = x + (w - draw_w) / 2
    draw_y = y + (h - draw_h) / 2

    c.setFillColor(PDF_PANEL_2)
    c.setStrokeColor(PDF_BORDER)
    c.roundRect(x, y, w, h, 10, stroke=1, fill=1)

    c.drawImage(reader, draw_x, draw_y, width=draw_w, height=draw_h, mask="auto")

    if caption:
        c.setFillColor(PDF_MUTED)
        c.setFont("Helvetica", 8)
        c.drawCentredString(x + w / 2, y - 12, caption)


def draw_figure_fit(c, fig, x, y, w, h, caption=None):
    reader, (img_w, img_h) = figure_to_reader(fig)

    scale = min(w / img_w, h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale

    draw_x = x + (w - draw_w) / 2
    draw_y = y + (h - draw_h) / 2

    c.setFillColor(PDF_PANEL_2)
    c.setStrokeColor(PDF_BORDER)
    c.roundRect(x, y, w, h, 10, stroke=1, fill=1)

    c.drawImage(reader, draw_x, draw_y, width=draw_w, height=draw_h, mask="auto")

    if caption:
        c.setFillColor(PDF_MUTED)
        c.setFont("Helvetica", 8)
        c.drawCentredString(x + w / 2, y - 12, caption)


def draw_note_box(c, x, y, w, h, title, body):
    c.setFillColor(PDF_PANEL)
    c.setStrokeColor(PDF_BORDER)
    c.roundRect(x, y, w, h, 10, stroke=1, fill=1)

    c.setFillColor(PDF_WHITE)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 12, y + h - 18, title)

    draw_wrapped_text(
        c,
        body,
        x + 12,
        y + h - 36,
        w - 24,
        font_name="Helvetica",
        font_size=8.5,
        leading=12,
        color=PDF_MUTED
    )


# -----------------------------
# PDF report generator
# -----------------------------
def build_pdf_report(result):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    page = 1

    draw_pdf_background(c)
    draw_footer(c, page)

    # -----------------------------
    # Page 1: Overview
    # -----------------------------
    draw_title(
        c,
        "AI Image Detection Report",
        "Deep learning prediction, digital forensics, test-time stability, and explainable visual analysis."
    )

    verdict_color = pdf_color_for_verdict(result["final_verdict"])

    # Main verdict box
    c.setFillColor(PDF_PANEL)
    c.setStrokeColor(verdict_color)
    c.setLineWidth(1.2)
    c.roundRect(40, height - 132, width - 80, 44, 12, stroke=1, fill=1)

    c.setFillColor(PDF_MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(55, height - 106, "FINAL VERDICT")

    c.setFillColor(verdict_color)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(55, height - 127, result["final_verdict"])

    # Horizontal metric cards
    card_y = height - 210
    card_h = 58
    gap = 8
    card_w = (width - 80 - (gap * 4)) / 5

    draw_card(c, 40 + (card_w + gap) * 0, card_y, card_w, card_h, "CNN AI Score", f"{result['ai_probability']:.2f}%", PDF_AI)
    draw_card(c, 40 + (card_w + gap) * 1, card_y, card_w, card_h, "Real Score", f"{result['real_probability']:.2f}%", PDF_REAL)
    draw_card(c, 40 + (card_w + gap) * 2, card_y, card_w, card_h, "Forensics", f"{result['digital_forensics_score']:.2f}%", PDF_WARN)
    draw_card(c, 40 + (card_w + gap) * 3, card_y, card_w, card_h, "Fusion", f"{result['fusion_score']:.2f}%", verdict_color)
    draw_card(c, 40 + (card_w + gap) * 4, card_y, card_w, card_h, "Confidence", result["confidence_level"], PDF_WHITE, value_size=13)

    # Image and summary
    draw_section_title(c, "Analyzed Image", 40, height - 250)
    draw_image_fit(c, result["image"], 40, height - 560, 230, 290, "Uploaded Image")

    draw_section_title(c, "Summary", 300, height - 250)

    summary_text = (
        "This report combines EfficientNet CNN classification with digital forensic indicators "
        "including metadata, noise, frequency, edge, color, and texture analysis. Test-time "
        "augmentation is used to estimate prediction stability. When CNN and forensic evidence "
        "disagree, the result should be interpreted carefully."
    )

    y_text = height - 275
    y_text = draw_wrapped_text(c, summary_text, 300, y_text, 245, font_size=9, leading=13, color=PDF_TEXT)

    y_text -= 12
    draw_wrapped_text(c, f"Portrait Adjustment: {result['portrait_note']}", 300, y_text, 245, font_size=8.5, leading=12, color=PDF_MUTED)

    # Horizontal forensic cards
    draw_section_title(c, "Digital Forensics Components", 40, 245)

    comp_y_1 = 178
    comp_y_2 = 108
    comp_w = (width - 80 - (gap * 2)) / 3

    draw_card(c, 40 + (comp_w + gap) * 0, comp_y_1, comp_w, 50, "Noise", f"{result['forensic_components']['noise_component']}%", PDF_WHITE, value_size=14)
    draw_card(c, 40 + (comp_w + gap) * 1, comp_y_1, comp_w, 50, "Frequency", f"{result['forensic_components']['frequency_component']}%", PDF_WHITE, value_size=14)
    draw_card(c, 40 + (comp_w + gap) * 2, comp_y_1, comp_w, 50, "Edge", f"{result['forensic_components']['edge_component']}%", PDF_WHITE, value_size=14)

    draw_card(c, 40 + (comp_w + gap) * 0, comp_y_2, comp_w, 50, "Metadata", f"{result['forensic_components']['metadata_component']}%", PDF_WHITE, value_size=14)
    draw_card(c, 40 + (comp_w + gap) * 1, comp_y_2, comp_w, 50, "Color", f"{result['forensic_components']['color_component']}%", PDF_WHITE, value_size=14)
    draw_card(c, 40 + (comp_w + gap) * 2, comp_y_2, comp_w, 50, "Texture", f"{result['forensic_components']['texture_component']}%", PDF_WHITE, value_size=14)

    # -----------------------------
    # Page 2: Grad-CAM
    # -----------------------------
    page += 1
    new_pdf_page(c, page)

    draw_title(c, "Explainable AI Visualization", "Grad-CAM heatmap highlights regions that influenced the CNN prediction.")

    draw_image_fit(c, result["heatmap_img"], 45, height - 420, 240, 285, "Grad-CAM Heatmap")
    draw_image_fit(c, result["overlay_img"], 310, height - 420, 240, 285, "Heatmap Overlay")

    draw_note_box(
        c,
        45,
        105,
        505,
        120,
        "Interpretation Note",
        "The heatmap is not a final proof of authenticity. It shows where the CNN focused during "
        "classification. The final system verdict should be interpreted using the CNN score, forensic "
        "score, fusion score, and uncertainty together."
    )

    # -----------------------------
    # Page 3: Forensic visual maps
    # -----------------------------
    page += 1
    new_pdf_page(c, page)

    draw_title(c, "Forensic Visual Analysis", "Visual forensic maps used as supporting evidence.")

    draw_image_fit(c, result["ela"]["ela_image"], 40, height - 335, 160, 190, "ELA Map")
    draw_image_fit(c, result["frequency"]["spectrum_image"], 218, height - 335, 160, 190, "Frequency Spectrum")
    draw_image_fit(c, result["edges"]["edge_image"], 396, height - 335, 160, 190, "Edge Map")

    draw_image_fit(c, result["color"]["color_map"], 80, height - 610, 200, 190, "Color Map")
    draw_image_fit(c, result["texture"]["texture_map"], 315, height - 610, 200, 190, "Texture Map")

    # -----------------------------
    # Page 4: Detailed forensic notes
    # -----------------------------
    page += 1
    new_pdf_page(c, page)

    draw_title(c, "Detailed Analysis Notes", "Metadata and forensic module outputs.")

    left_x = 40
    right_x = 310
    y_left = height - 110
    y_right = height - 110

    draw_section_title(c, "Metadata", left_x, y_left)
    y_left -= 22

    metadata_lines = [
        f"EXIF Present: {result['metadata']['has_exif']}",
        f"Camera Make: {result['metadata']['camera_make']}",
        f"Camera Model: {result['metadata']['camera_model']}",
        f"Lens Model: {result['metadata']['lens_model']}",
        f"Date/Time: {result['metadata']['datetime']}",
        f"Software: {result['metadata']['software']}",
        f"GPS Present: {result['metadata']['gps_info']}",
        f"Orientation: {result['metadata']['orientation']}",
        f"Forensic Note: {result['metadata']['forensic_note']}",
    ]

    for item in metadata_lines:
        y_left = draw_wrapped_text(c, item, left_x, y_left, 240, font_size=8.5, leading=12, color=PDF_TEXT)
        y_left -= 3

    draw_section_title(c, "Forensic Scores", right_x, y_right)
    y_right -= 22

    forensic_lines = [
        f"Noise Level: {result['noise']['noise_value']}",
        f"Noise Note: {result['noise']['forensic_note']}",
        f"ELA Score: {result['ela']['ela_score']}",
        f"ELA Note: {result['ela']['forensic_note']}",
        f"Frequency Score: {result['frequency']['frequency_score']}",
        f"Frequency Note: {result['frequency']['forensic_note']}",
        f"Edge Score: {result['edges']['edge_score']}",
        f"Edge Note: {result['edges']['forensic_note']}",
        f"Color Score: {result['color']['color_score']}",
        f"Texture Score: {result['texture']['texture_score']}",
        f"Texture Note: {result['texture']['forensic_note']}",
    ]

    for item in forensic_lines:
        y_right = draw_wrapped_text(c, item, right_x, y_right, 240, font_size=8.5, leading=12, color=PDF_TEXT)
        y_right -= 3

    draw_note_box(
        c,
        40,
        55,
        515,
        75,
        "Report Disclaimer",
        "This system provides an AI-likelihood assessment based on trained data and forensic indicators. "
        "Highly realistic AI images, edited real photos, compression, and missing metadata may affect results."
    )

    c.save()
    buffer.seek(0)
    return buffer


# -----------------------------
# Improved verdict helper with uncertainty fix
# -----------------------------
def get_verdict_and_confidence(fusion_score, tta_uncertainty=0):
    if fusion_score >= 60:
        final_verdict = "Likely AI Generated"
        verdict_class = "result-ai"

    elif fusion_score >= 35 or tta_uncertainty >= 8:
        final_verdict = "Uncertain / Needs Review"
        verdict_class = "result-warn"

    else:
        final_verdict = "Likely Real Photograph"
        verdict_class = "result-real"

    if tta_uncertainty >= 8:
        confidence_level = "Low"
    else:
        distance_from_center = abs(fusion_score - 50)

        if distance_from_center >= 35:
            confidence_level = "Very High"
        elif distance_from_center >= 25:
            confidence_level = "High"
        elif distance_from_center >= 15:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

    return final_verdict, verdict_class, confidence_level


# -----------------------------
# Portrait false-positive safety
# -----------------------------
def apply_portrait_safety_adjustment(image, ai_probability, fusion_score):
    width, height = image.size
    aspect_ratio = width / max(height, 1)

    portrait_like = 0.7 <= aspect_ratio <= 1.3

    adjusted_score = fusion_score
    note = "No portrait adjustment applied."

    if portrait_like and ai_probability > 95:
        adjusted_score = max(0, fusion_score - 5)
        note = "Small portrait adjustment applied to reduce possible false positives."

    return adjusted_score, note


# -----------------------------
# Main analysis
# -----------------------------
def process_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")

    probabilities, tta_std = predict_with_tta(model, image)
    tta_uncertainty = float(tta_std[0][0].item() * 100)

    ai_probability_raw = probabilities[0][0].item()
    real_probability_raw = probabilities[0][1].item()

    ai_probability = ai_probability_raw * 100
    real_probability = real_probability_raw * 100

    confidence, predicted = torch.max(probabilities, 1)
    confidence_percent = confidence.item() * 100
    prediction_label = classes[predicted.item()]

    metadata = analyze_metadata(image)
    noise = analyze_noise(image)
    ela = analyze_ela(image)
    frequency = analyze_frequency(image)
    edges = analyze_edges(image)
    color = analyze_color(image)
    texture = analyze_texture(image)

    digital_forensics_score, forensic_components = calculate_digital_forensics_score(
        noise_score=noise["noise_value"],
        freq_score=frequency["frequency_score"],
        edge_score=edges["edge_score"],
        metadata_flag=metadata["has_exif"],
        color_score=color["color_score"],
        texture_score=texture["texture_score"]
    )

    fusion_score = compute_final_score(
        cnn_prob=ai_probability_raw,
        noise_score=noise["noise_value"],
        freq_score=frequency["frequency_score"],
        edge_score=edges["edge_score"],
        metadata_flag=metadata["has_exif"],
        color_score=color["color_score"],
        texture_score=texture["texture_score"]
    )

    fusion_score, portrait_note = apply_portrait_safety_adjustment(
        image,
        ai_probability,
        fusion_score
    )

    final_verdict, verdict_class, confidence_level = get_verdict_and_confidence(
        fusion_score,
        tta_uncertainty
    )

    if prediction_label == "ai":
        prediction_text = "AI Generated Image"
        prediction_class = "result-ai"
    else:
        prediction_text = "Real Photograph"
        prediction_class = "result-real"

    input_tensor = image_to_tensor(image)
    target_layer = model.features[-1]
    cam = generate_gradcam(model, input_tensor, predicted.item(), target_layer)
    heatmap_img, overlay_img = create_heatmap_overlay(image, cam)

    radar_fig = create_radar_chart(
        cnn_ai=ai_probability,
        noise_score=noise["noise_value"],
        freq_score=frequency["frequency_score"],
        edge_score=edges["edge_score"],
        metadata_flag=metadata["has_exif"],
        color_score=color["color_score"],
        texture_score=texture["texture_score"]
    )

    return {
        "image": image,
        "prediction_label": prediction_label,
        "prediction_text": prediction_text,
        "prediction_class": prediction_class,
        "confidence_percent": confidence_percent,
        "ai_probability": ai_probability,
        "real_probability": real_probability,
        "tta_uncertainty": tta_uncertainty,
        "digital_forensics_score": digital_forensics_score,
        "forensic_components": forensic_components,
        "metadata": metadata,
        "noise": noise,
        "ela": ela,
        "frequency": frequency,
        "edges": edges,
        "color": color,
        "texture": texture,
        "fusion_score": fusion_score,
        "confidence_level": confidence_level,
        "final_verdict": final_verdict,
        "verdict_class": verdict_class,
        "portrait_note": portrait_note,
        "heatmap_img": heatmap_img,
        "overlay_img": overlay_img,
        "radar_fig": radar_fig,
    }


# -----------------------------
# Session state
# -----------------------------
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# -----------------------------
# Homepage
# -----------------------------
if st.session_state.analysis_result is None:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-icon">🛡️</div>
        <div class="hero-title">AI Image Detection System</div>
        <div class="hero-subtitle">
            Advanced deep learning and forensic analysis for identifying AI-generated images.
            Upload an image to inspect authenticity using CNN prediction, digital forensic signals,
            test-time stability, and explainable visual analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 2.6, 1])

    with center:
        st.markdown('<div class="upload-shell">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        st.markdown(
            '<div class="upload-box-label">Drag and drop an image here, or click Browse files</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            loading_placeholder = st.empty()
            progress_placeholder = st.empty()

            loading_placeholder.markdown("""
            <div class="loading-box">
                <div class="loading-title">Analyzing Image</div>
                <div class="loading-subtitle">
                    Running CNN prediction, test-time augmentation, metadata inspection,
                    noise analysis, error level analysis, frequency analysis, edge analysis,
                    color analysis, texture analysis, Grad-CAM visualization,
                    digital forensics scoring, and final fusion scoring...
                </div>
            </div>
            """, unsafe_allow_html=True)

            progress = progress_placeholder.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            try:
                st.session_state.analysis_result = process_uploaded_image(uploaded_file)
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

else:
    result = st.session_state.analysis_result
    pdf_buffer = build_pdf_report(result)

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="section-label">Analyzed Image</div>', unsafe_allow_html=True)
        st.image(result["image"], use_container_width=True)

    with right:
        st.markdown('<div class="card-title">Prediction Overview</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="result-box {result["prediction_class"]}">{result["prediction_text"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="panel">
            <div class="small-label">CNN AI Likelihood</div>
            <div class="big-score">{result["ai_probability"]:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(int(result["ai_probability"]), 100))

        st.markdown(f"""
        <div class="panel">
            <div class="small-label">Digital Forensics Score</div>
            <div class="big-score">{result["digital_forensics_score"]:.2f}%</div>
            <div class="small-label">
                Combined score from metadata, noise, frequency, edge, color, and texture analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(int(result["digital_forensics_score"]), 100))

        st.markdown(
            f'<div class="result-box {result["verdict_class"]}">{result["final_verdict"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="panel">
            <div class="small-label">Final Fused AI Score</div>
            <div class="big-score">{result["fusion_score"]:.2f}%</div>
            <div class="small-label">Confidence Level: {result["confidence_level"]}</div>
            <div class="small-label">TTA Uncertainty: {result["tta_uncertainty"]:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="summary-title">Analysis Summary</div>', unsafe_allow_html=True)
        summary_text = (
            "The final verdict combines a trained EfficientNet model with a digital forensics score. "
            "The digital forensics score is calculated from metadata, noise, frequency, edge, color, "
            "and texture indicators. Test-time augmentation is also used to measure prediction stability. "
            "Images with unstable predictions are marked as uncertain instead of being confidently classified as real."
        )
        st.markdown(f'<div class="summary-text">{summary_text}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="note-box">
            Missing metadata alone is not treated as strong evidence of AI generation,
            because many real images from websites and social media lose EXIF information.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="note-box">
            {result["portrait_note"]}
        </div>
        """, unsafe_allow_html=True)

        btn1, btn2 = st.columns(2)

        with btn1:
            if st.button("Upload Another Image", use_container_width=True):
                st.session_state.analysis_result = None
                st.rerun()

        with btn2:
            st.download_button(
                "Download Result",
                data=pdf_buffer,
                file_name="ai_detection_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card-title">Detection Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="medium")

    with c1:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">CNN Prediction</div>'
            f'<div class="metric-value">{result["prediction_label"].upper()}</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">CNN AI</div>'
            f'<div class="metric-value">{result["ai_probability"]:.2f}%</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Digital Forensics</div>'
            f'<div class="metric-value">{result["digital_forensics_score"]:.2f}%</div></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Fusion Score</div>'
            f'<div class="metric-value">{result["fusion_score"]:.2f}%</div></div>',
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Confidence</div>'
            f'<div class="metric-value">{result["confidence_level"]}</div></div>',
            unsafe_allow_html=True
        )

    with c6:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">TTA Uncertainty</div>'
            f'<div class="metric-value">{result["tta_uncertainty"]:.2f}%</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card-title">Forensic Radar Chart</div>', unsafe_allow_html=True)
    st.pyplot(result["radar_fig"], use_container_width=False)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Digital Forensics",
        "Metadata Analysis",
        "Noise Analysis",
        "ELA Analysis",
        "Frequency Analysis",
        "Edge Analysis",
        "Color Analysis",
        "Texture Analysis",
        "AI Heatmap"
    ])

    with tab1:
        st.write(f"**Digital Forensics Score:** {result['digital_forensics_score']:.2f}%")
        st.write("This score combines the supporting forensic indicators below.")
        st.write(f"**Noise Component:** {result['forensic_components']['noise_component']}%")
        st.write(f"**Frequency Component:** {result['forensic_components']['frequency_component']}%")
        st.write(f"**Edge Component:** {result['forensic_components']['edge_component']}%")
        st.write(f"**Metadata Component:** {result['forensic_components']['metadata_component']}%")
        st.write(f"**Color Component:** {result['forensic_components']['color_component']}%")
        st.write(f"**Texture Component:** {result['forensic_components']['texture_component']}%")

    with tab2:
        st.write(f"**EXIF Metadata Present:** {result['metadata']['has_exif']}")
        st.write(f"**Camera Make:** {result['metadata']['camera_make']}")
        st.write(f"**Camera Model:** {result['metadata']['camera_model']}")
        st.write(f"**Lens Model:** {result['metadata']['lens_model']}")
        st.write(f"**Date/Time:** {result['metadata']['datetime']}")
        st.write(f"**Software:** {result['metadata']['software']}")
        st.write(f"**Artist:** {result['metadata']['artist']}")
        st.write(f"**Copyright:** {result['metadata']['copyright']}")
        st.write(f"**Image Description:** {result['metadata']['image_description']}")
        st.write(f"**Host Computer:** {result['metadata']['host_computer']}")
        st.write(f"**GPS Info Present:** {result['metadata']['gps_info']}")
        st.write(f"**Orientation:** {result['metadata']['orientation']}")
        st.write(f"**X Resolution:** {result['metadata']['x_resolution']}")
        st.write(f"**Y Resolution:** {result['metadata']['y_resolution']}")
        st.write(f"**Forensic Note:** {result['metadata']['forensic_note']}")

    with tab3:
        st.write(f"**Noise Level:** {result['noise']['noise_value']}")
        st.write(f"**Forensic Note:** {result['noise']['forensic_note']}")

    with tab4:
        st.write(f"**ELA Score:** {result['ela']['ela_score']}")
        st.write(f"**Forensic Note:** {result['ela']['forensic_note']}")
        st.image(result["ela"]["ela_image"], caption="Error Level Analysis (ELA)", use_container_width=True)

    with tab5:
        st.write(f"**Frequency Score:** {result['frequency']['frequency_score']}")
        st.write(f"**Forensic Note:** {result['frequency']['forensic_note']}")
        st.image(result["frequency"]["spectrum_image"], caption="FFT Magnitude Spectrum", use_container_width=True)

    with tab6:
        st.write(f"**Edge Score:** {result['edges']['edge_score']}")
        st.write(f"**Forensic Note:** {result['edges']['forensic_note']}")
        st.image(result["edges"]["edge_image"], caption="Edge Consistency Map", use_container_width=True)

    with tab7:
        st.write(f"**Color Score:** {result['color']['color_score']}")
        st.write(f"**L Channel Std:** {result['color']['l_std']}")
        st.write(f"**A Channel Std:** {result['color']['a_std']}")
        st.write(f"**B Channel Std:** {result['color']['b_std']}")
        st.write(f"**Forensic Note:** {result['color']['forensic_note']}")
        st.image(result["color"]["color_map"], caption="Color Inconsistency Map", use_container_width=True)

    with tab8:
        st.write(f"**Texture Score:** {result['texture']['texture_score']}")
        st.write(f"**Forensic Note:** {result['texture']['forensic_note']}")
        st.image(result["texture"]["texture_map"], caption="Texture Consistency Map", use_container_width=True)

    with tab9:
        st.write("This visualization highlights the image regions used by the CNN during classification.")
        col_a, col_b = st.columns(2)

        with col_a:
            st.image(result["heatmap_img"], caption="Grad-CAM Heatmap", use_container_width=True)

        with col_b:
            st.image(result["overlay_img"], caption="Heatmap Overlay", use_container_width=True)
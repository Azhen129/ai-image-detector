# AI Image Detection System

An AI-generated image detection application developed using deep learning and digital forensic analysis.

The system uses a trained **EfficientNet-B0 convolutional neural network (CNN)** together with forensic indicators to assess whether an uploaded image is likely AI-generated or a real photograph.

## Features

- **AI Image Classification:** Classifies uploaded images using a trained EfficientNet-B0 model.
- **Digital Forensics:** Examines image metadata, noise, frequency, edges, color, and texture.
- **Error Level Analysis (ELA):** Displays compression-related differences in images.
- **Test-Time Augmentation (TTA):** Tests multiple image variations to measure prediction stability.
- **Fusion Scoring:** Combines CNN predictions with forensic indicators for a final assessment.
- **Grad-CAM Visualization:** Highlights image regions that influenced the CNN prediction.
- **Interactive Dashboard:** Built with Streamlit for image uploading and result exploration.
- **PDF Reports:** Generates downloadable reports containing detection results and forensic analysis.

## Technologies Used

Python, PyTorch, Torchvision, EfficientNet-B0, Streamlit, OpenCV, NumPy, Pillow, Matplotlib, ReportLab, and scikit-image.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/ai-image-detector.git
cd ai-image-detector
```

Replace `YOUR_USERNAME` with your GitHub username.

**2. Create a virtual environment**

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**4. Run the application**

```bash
python -m streamlit run app/app.py
```

The Streamlit application will open in your web browser.

## How It Works

1. Upload a JPG, JPEG, or PNG image.
2. The EfficientNet-B0 model predicts whether the image is AI-generated or real.
3. The system analyzes supporting digital forensic indicators.
4. Test-time augmentation measures the stability of the CNN prediction.
5. The fusion system calculates a final score and verdict.
6. Explore the forensic visualizations and download a PDF analysis report.

## Project Structure

```text
ai-image-detector/
├── app/
│   └── app.py
├── models/
│   └── final_model.pt
├── src/
│   ├── forensics/
│   ├── fusion/
│   └── model/
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

The system provides an AI-likelihood assessment rather than definitive proof of image authenticity.

Highly realistic AI images, edited photographs, compression, and missing metadata can affect detection results. Forensic scores are supporting indicators and should not be interpreted as independently calibrated probabilities.

## Project Background

Developed as a university-level computer science and cybersecurity project, combining deep learning, digital image forensics, and explainable AI techniques.
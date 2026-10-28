import os
import io
import copy
import random
from collections import Counter

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# Project paths
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # src/model
SRC_DIR = os.path.dirname(CURRENT_DIR)                        # src
PROJECT_DIR = os.path.dirname(SRC_DIR)                        # project root

DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pt")
BACKUP_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_previous.pt")

print("\n--- PATH CHECK ---")
print("CURRENT_DIR:", CURRENT_DIR)
print("SRC_DIR:", SRC_DIR)
print("PROJECT_DIR:", PROJECT_DIR)
print("DATA_DIR:", DATA_DIR)
print("TRAIN_DIR:", TRAIN_DIR)
print("VAL_DIR:", VAL_DIR)
print("TEST_DIR:", TEST_DIR)
print("BEST_MODEL_PATH:", BEST_MODEL_PATH)
print("FINAL_MODEL_PATH:", FINAL_MODEL_PATH)
print("------------------\n")


# -----------------------------
# Backup old model before overwriting
# -----------------------------
if os.path.exists(BEST_MODEL_PATH) and not os.path.exists(BACKUP_MODEL_PATH):
    import shutil
    shutil.copy2(BEST_MODEL_PATH, BACKUP_MODEL_PATH)
    print(f"Backup created: {BACKUP_MODEL_PATH}")


# -----------------------------
# Validate dataset folders
# -----------------------------
required_dirs = [
    DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    os.path.join(TRAIN_DIR, "ai"),
    os.path.join(TRAIN_DIR, "real"),
    os.path.join(VAL_DIR, "ai"),
    os.path.join(VAL_DIR, "real"),
    os.path.join(TEST_DIR, "ai"),
    os.path.join(TEST_DIR, "real"),
]

missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]

if missing_dirs:
    print("Missing folders:")
    for d in missing_dirs:
        print(d)
    raise FileNotFoundError("Some required dataset folders do not exist.")


# -----------------------------
# Check dataset image files
# -----------------------------
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_paths(folder):
    paths = []

    for root, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()

            if ext in VALID_EXTENSIONS:
                paths.append(os.path.join(root, file))

    return paths


def verify_images(folder_name, folder_path):
    image_paths = collect_image_paths(folder_path)

    print(f"{folder_name}: {len(image_paths)} image files found")

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image files found in {folder_path}")

    bad_files = []

    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            bad_files.append((path, str(e)))

    if bad_files:
        print(f"\nBad files found in {folder_name}:")
        for path, err in bad_files[:20]:
            print(f"{path} -> {err}")

        raise FileNotFoundError(f"Corrupt or unreadable image files found in {folder_name}")

    return image_paths


print("--- VERIFYING IMAGES ---")
verify_images("TRAIN", TRAIN_DIR)
verify_images("VAL", VAL_DIR)
verify_images("TEST", TEST_DIR)
print("------------------------\n")


# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 32
num_epochs = 20
learning_rate_head = 1e-3
learning_rate_finetune = 5e-5
weight_decay = 1e-4
num_classes = 2
early_stopping_patience = 5
unfreeze_epoch = 3
num_workers = 0

# Balanced threshold.
# Class index 0 = AI
# Class index 1 = Real
AI_DECISION_THRESHOLD = 0.50


# -----------------------------
# Custom augmentation helpers
# -----------------------------
class RandomJPEGCompression:
    def __init__(self, p=0.35, quality_range=(55, 95)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() > self.p:
            return img

        buffer = io.BytesIO()
        quality = random.randint(*self.quality_range)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        compressed = Image.open(buffer).convert("RGB")
        return compressed


class AddGaussianNoise:
    def __init__(self, p=0.25, std_range=(1, 6)):
        self.p = p
        self.std_range = std_range

    def __call__(self, img):
        if random.random() > self.p:
            return img

        arr = np.array(img).astype(np.float32)
        std = random.uniform(*self.std_range)
        noise = np.random.normal(0, std, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(arr)


# -----------------------------
# Normalization values
# -----------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# -----------------------------
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
    RandomJPEGCompression(p=0.35, quality_range=(55, 95)),
    AddGaussianNoise(p=0.25, std_range=(1, 6)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


# -----------------------------
# Datasets
# -----------------------------
print("Creating datasets...")

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)

print("Classes:", train_dataset.classes)
print("Class to index:", train_dataset.class_to_idx)
print("Train images:", len(train_dataset))
print("Val images:", len(val_dataset))
print("Test images:", len(test_dataset))


# -----------------------------
# Class balancing
# -----------------------------
train_targets = [label for _, label in train_dataset.samples]
class_counts = Counter(train_targets)

print("Training class counts:", dict(class_counts))

sample_weights = [1.0 / class_counts[label] for label in train_targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# FINAL POLISHING SETTING:
# 1.8 was too aggressive and made real images look AI.
# 1.1 was too weak and missed too many AI images.
# 1.3 worked well, but for hard real portraits, 1.2 is safer.
class_weights = torch.tensor([1.2, 1.0], dtype=torch.float32).to(device)

print("Loss class weights:", class_weights.cpu().numpy())


# -----------------------------
# DataLoaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)


# -----------------------------
# Model
# -----------------------------
print("\nCreating EfficientNet model...")

try:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    print("Loaded pretrained EfficientNet_B0 weights.")
except Exception as e:
    print(f"Could not load pretrained weights: {e}")
    print("Falling back to random initialization.")
    model = efficientnet_b0(weights=None)

for param in model.features.parameters():
    param.requires_grad = False

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

model = model.to(device)


# -----------------------------
# Loss / Optimizer / Scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate_head,
    weight_decay=weight_decay
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# -----------------------------
# Metrics
# -----------------------------
def predict_with_threshold(probs, threshold=AI_DECISION_THRESHOLD):
    """
    Class index 0 = AI
    Class index 1 = Real

    If AI probability is >= threshold, predict AI.
    Otherwise, predict real.
    """

    ai_probs = probs[:, 0]

    preds = torch.where(
        ai_probs >= threshold,
        torch.zeros_like(ai_probs, dtype=torch.long),
        torch.ones_like(ai_probs, dtype=torch.long)
    )

    return preds


def compute_metrics(y_true, y_pred, y_prob_ai):
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    try:
        y_true_binary_ai = [1 if y == 0 else 0 for y in y_true]
        auc = roc_auc_score(y_true_binary_ai, y_prob_ai)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": accuracy,
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": f1,
        "auc_ai": auc,
        "confusion_matrix": cm
    }


def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []
    y_prob_ai = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = predict_with_threshold(probs)

            running_loss += loss.item()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob_ai.extend(probs[:, 0].cpu().numpy().tolist())

    avg_loss = running_loss / len(loader)

    metrics = compute_metrics(y_true, y_pred, y_prob_ai)
    metrics["loss"] = avg_loss

    return metrics


# -----------------------------
# Training loop
# -----------------------------
best_val_f1 = 0.0
best_val_accuracy = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

print("\nStarting hard-real polishing training...\n")
print(f"AI decision threshold: {AI_DECISION_THRESHOLD}")
print("Goal: keep strong AI detection while reducing false positives on real portraits.\n")

for epoch in range(num_epochs):
    model.train()

    if epoch == unfreeze_epoch:
        print("\nUnfreezing EfficientNet backbone for fine-tuning...\n")

        for param in model.features.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate_finetune,
            weight_decay=weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )

    running_loss = 0.0
    y_true_train = []
    y_pred_train = []
    y_prob_ai_train = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        probs = torch.softmax(outputs, dim=1)
        preds = predict_with_threshold(probs)

        running_loss += loss.item()

        y_true_train.extend(labels.cpu().numpy().tolist())
        y_pred_train.extend(preds.cpu().numpy().tolist())
        y_prob_ai_train.extend(probs[:, 0].detach().cpu().numpy().tolist())

    train_loss = running_loss / len(train_loader)

    train_metrics = compute_metrics(
        y_true_train,
        y_pred_train,
        y_prob_ai_train
    )

    train_metrics["loss"] = train_loss

    val_metrics = evaluate(model, val_loader, criterion)
    scheduler.step(val_metrics["f1_ai"])

    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    print(
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train Acc: {train_metrics['accuracy'] * 100:.2f}% | "
        f"Train Prec(AI): {train_metrics['precision_ai']:.4f} | "
        f"Train Rec(AI): {train_metrics['recall_ai']:.4f} | "
        f"Train F1(AI): {train_metrics['f1_ai']:.4f} | "
        f"Train AUC(AI): {train_metrics['auc_ai']:.4f}"
    )

    print(
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
        f"Val Prec(AI): {val_metrics['precision_ai']:.4f} | "
        f"Val Rec(AI): {val_metrics['recall_ai']:.4f} | "
        f"Val F1(AI): {val_metrics['f1_ai']:.4f} | "
        f"Val AUC(AI): {val_metrics['auc_ai']:.4f}"
    )

    print("Val Confusion Matrix [rows=true, cols=pred]:")
    print(val_metrics["confusion_matrix"])
    print("-" * 70)

    # Balanced model selection:
    # Save by F1 because F1 balances precision and recall.
    improved = val_metrics["f1_ai"] > best_val_f1

    if improved:
        best_val_f1 = val_metrics["f1_ai"]
        best_val_accuracy = val_metrics["accuracy"]
        best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, BEST_MODEL_PATH)
        torch.save(best_model_wts, FINAL_MODEL_PATH)

        print("Best hard-real polished model saved.\n")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping triggered.")
        break


# -----------------------------
# Final test
# -----------------------------
model.load_state_dict(best_model_wts)

test_metrics = evaluate(model, test_loader, criterion)

print("\nTraining complete.")
print(f"Best Validation F1 (AI): {best_val_f1:.4f}")
print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")

print("\nFinal Test Results:")
print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
print(f"Test Precision (AI): {test_metrics['precision_ai']:.4f}")
print(f"Test Recall (AI): {test_metrics['recall_ai']:.4f}")
print(f"Test F1 (AI): {test_metrics['f1_ai']:.4f}")
print(f"Test AUC (AI): {test_metrics['auc_ai']:.4f}")
print("Test Confusion Matrix [rows=true, cols=pred]:")
print(test_metrics["confusion_matrix"])

print(f"\nBest model saved to: {BEST_MODEL_PATH}")
print(f"Final locked model saved to: {FINAL_MODEL_PATH}")
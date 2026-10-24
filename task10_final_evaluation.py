
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import functional as TF

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from task2_image_pipeline import (
    FacialFeatureDataset,
    val_transform,
    class_to_idx,
)

# ==========================================================
# TASK 10 - FINAL ENSEMBLE EVALUATION
# ==========================================================

project = Path(__file__).resolve().parent
results = project / "results"
results.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16

print("Device:", device, flush=True)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

class_names = [
    name for name, index in sorted(
        class_to_idx.items(),
        key=lambda item: item[1]
    )
]

# ==========================================================
# 1. LOAD FINAL TRAINED MODELS
# ==========================================================

resnet = models.resnet18(weights=None)

resnet.fc = nn.Linear(
    resnet.fc.in_features,
    len(class_to_idx)
)

resnet.load_state_dict(
    torch.load(
        project / "best_resnet18_finetuned.pth",
        map_location="cpu",
        weights_only=True
    )
)

resnet = resnet.to(device)
resnet.eval()

efficientnet = models.efficientnet_b0(weights=None)

efficientnet.classifier[1] = nn.Linear(
    efficientnet.classifier[1].in_features,
    len(class_to_idx)
)

efficientnet.load_state_dict(
    torch.load(
        project / "best_efficientnet_b0.pth",
        map_location="cpu",
        weights_only=True
    )
)

efficientnet = efficientnet.to(device)
efficientnet.eval()

print("Both models loaded successfully!", flush=True)

# ==========================================================
# 2. LOAD VALIDATION DATASET
# ==========================================================

val_df = pd.read_csv(
    results / "validation_split.csv"
)

val_dataset = FacialFeatureDataset(
    val_df,
    project / "train",
    val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print("Validation images:", len(val_dataset), flush=True)

# ==========================================================
# 3. TEST-TIME AUGMENTATION
# ==========================================================

def predict_tta(model, images):

    original_probs = torch.softmax(
        model(images),
        dim=1
    )

    flipped_probs = torch.softmax(
        model(TF.hflip(images)),
        dim=1
    )

    return (original_probs + flipped_probs) / 2

# ==========================================================
# 4. EVALUATE FINAL 50/50 ENSEMBLE
# ==========================================================

all_predictions = []
all_labels = []

print("\nEvaluating final ensemble...", flush=True)

with torch.inference_mode():

    for batch_number, (images, labels) in enumerate(
        val_loader,
        start=1
    ):

        images = images.to(device)

        resnet_probs = predict_tta(
            resnet,
            images
        )

        efficientnet_probs = predict_tta(
            efficientnet,
            images
        )

        combined_probs = (
            0.5 * resnet_probs
            + 0.5 * efficientnet_probs
        )

        predictions = combined_probs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        if batch_number % 200 == 0:
            print(
                f"Processed {batch_number}/{len(val_loader)} batches",
                flush=True
            )

# ==========================================================
# 5. OVERALL ACCURACY
# ==========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\nFINAL ENSEMBLE RESULTS")
print(f"Validation accuracy: {accuracy:.2%}")

# ==========================================================
# 6. CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nClassification report:")
print(report)

report_path = (
    results / "final_ensemble_classification_report.txt"
)

report_path.write_text(
    report,
    encoding="utf-8"
)

report_dict = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

report_df.to_csv(
    results / "final_ensemble_per_class_metrics.csv"
)

# ==========================================================
# 7. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    results / "final_ensemble_confusion_matrix.csv"
)

fig, ax = plt.subplots(figsize=(11, 9))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=True,
    xticks_rotation=45
)

ax.set_title(
    "Final Ensemble - Confusion Matrix",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    results / "final_ensemble_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# 8. PER-CLASS RECALL CHART
# ==========================================================

class_recall = [
    report_dict[name]["recall"] * 100
    for name in class_names
]

plt.figure(figsize=(11, 6))

bars = plt.bar(
    class_names,
    class_recall
)

plt.ylim(0, 105)

plt.ylabel("Recall (%)")
plt.xlabel("Class")

plt.title(
    "Final Ensemble - Recall by Class"
)

plt.xticks(rotation=35, ha="right")

for bar, value in zip(bars, class_recall):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    results / "final_ensemble_per_class_recall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# 9. FINAL SUMMARY
# ==========================================================

print("\nTASK 10 COMPLETED SUCCESSFULLY!")

print(
    "Confusion matrix:",
    results / "final_ensemble_confusion_matrix.png"
)

print(
    "Per-class chart:",
    results / "final_ensemble_per_class_recall.png"
)

print(
    "Classification report:",
    report_path
)

print(
    "\nFinal Kaggle public score: 97.59%"
)

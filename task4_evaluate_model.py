
# ENGE707 Assessment 3 - Task 4
# Evaluate the best ResNet18 competition model

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from task2_image_pipeline import (
    FacialFeatureDataset,
    val_transform,
    class_to_idx,
)

project = Path(__file__).resolve().parent
results = project / "results"
results.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Evaluation device:", device, flush=True)

# 1. Load training history
history = pd.read_csv(results / "training_history.csv")

# 2. Plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(
    history["epoch"],
    history["train_accuracy"] * 100,
    marker="o",
    label="Training accuracy",
)
plt.plot(
    history["epoch"],
    history["val_accuracy"] * 100,
    marker="o",
    label="Validation accuracy",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ResNet18 Training and Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(results / "accuracy_curve.png", dpi=200)
plt.close()

# 3. Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Training loss",
)
plt.plot(
    history["epoch"],
    history["val_loss"],
    marker="o",
    label="Validation loss",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet18 Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(results / "loss_curve.png", dpi=200)
plt.close()

# 4. Rebuild and load the best model
model = models.resnet18(weights=None)
model.fc = nn.Linear(
    model.fc.in_features,
    len(class_to_idx)
)

model.load_state_dict(
    torch.load(
        project / "best_resnet18.pth",
        map_location=device,
        weights_only=True,
    )
)

model = model.to(device)
model.eval()

# 5. Load validation images
val_df = pd.read_csv(results / "validation_split.csv")

val_dataset = FacialFeatureDataset(
    val_df,
    project / "train",
    val_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

# 6. Predict validation classes
true_labels = []
predicted_labels = []

print("Evaluating validation images...", flush=True)

with torch.inference_mode():
    for batch_number, (images, labels) in enumerate(
        val_loader, start=1
    ):
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        true_labels.extend(labels.tolist())
        predicted_labels.extend(
            predictions.cpu().tolist()
        )

        if batch_number % 200 == 0:
            print(
                f"Evaluated {batch_number}/{len(val_loader)} batches",
                flush=True,
            )

# 7. Classification report
class_names = [
    name for name, index in sorted(
        class_to_idx.items(),
        key=lambda item: item[1],
    )
]

report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print("\nCLASSIFICATION REPORT\n")
print(report)

with open(
    results / "classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report)

# 8. Confusion matrix
matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
)

fig, ax = plt.subplots(figsize=(11, 9))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names,
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=False,
)

plt.title("ResNet18 Validation Confusion Matrix")
plt.tight_layout()
plt.savefig(
    results / "confusion_matrix.png",
    dpi=200,
)
plt.close()

accuracy = (
    sum(
        actual == predicted
        for actual, predicted in zip(
            true_labels,
            predicted_labels,
        )
    )
    / len(true_labels)
)

print(f"\nValidation accuracy: {accuracy:.2%}")
print("Evaluation completed!")
print("Graphs and reports saved in results/")

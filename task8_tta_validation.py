
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import functional as TF

from task2_image_pipeline import (
    FacialFeatureDataset,
    val_transform,
    class_to_idx,
)

# Task 8A: Evaluate horizontal-flip test-time augmentation.

project = Path(__file__).resolve().parent
results = project / "results"
results.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

print("Device:", device, flush=True)

val_df = pd.read_csv(results / "validation_split.csv")

val_dataset = FacialFeatureDataset(
    val_df,
    project / "train",
    val_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))

model.load_state_dict(
    torch.load(
        project / "best_resnet18_finetuned.pth",
        map_location="cpu",
        weights_only=True,
    )
)

model = model.to(device)
model.eval()

normal_correct = 0
tta_correct = 0
total = 0

print("Validation images:", len(val_dataset), flush=True)
print("Evaluating original and flipped predictions...", flush=True)

with torch.inference_mode():
    for batch_number, (images, labels) in enumerate(val_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        # Standard prediction
        original_logits = model(images)
        original_probs = torch.softmax(original_logits, dim=1)

        # Horizontally flipped prediction
        flipped_images = TF.hflip(images)
        flipped_logits = model(flipped_images)
        flipped_probs = torch.softmax(flipped_logits, dim=1)

        # Average the probabilities
        tta_probs = (original_probs + flipped_probs) / 2

        normal_predictions = original_probs.argmax(dim=1)
        tta_predictions = tta_probs.argmax(dim=1)

        normal_correct += (normal_predictions == labels).sum().item()
        tta_correct += (tta_predictions == labels).sum().item()
        total += labels.size(0)

        if batch_number % 200 == 0:
            print(
                f"Processed {batch_number}/{len(val_loader)} batches",
                flush=True,
            )

normal_accuracy = normal_correct / total
tta_accuracy = tta_correct / total

comparison = pd.DataFrame([
    {
        "method": "Fine-tuned ResNet18",
        "validation_accuracy": normal_accuracy,
    },
    {
        "method": "Fine-tuned ResNet18 + Horizontal Flip TTA",
        "validation_accuracy": tta_accuracy,
    },
])

comparison.to_csv(results / "tta_validation_comparison.csv", index=False)

print("\nTask 8A completed!")
print(f"Normal validation accuracy: {normal_accuracy:.2%}")
print(f"TTA validation accuracy: {tta_accuracy:.2%}")
print(f"Difference: {(tta_accuracy - normal_accuracy) * 100:+.2f} percentage points")
print("Saved:", results / "tta_validation_comparison.csv")

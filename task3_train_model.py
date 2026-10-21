
# ENGE707 Assessment 3 - Task 3
# Kaggle competition: ResNet18 baseline training

from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from task2_image_pipeline import (
    FacialFeatureDataset,
    train_transform,
    val_transform,
    class_to_idx,
)

# -----------------------------------
# 1. Reproducibility
# -----------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------------------
# 2. Project folders and GPU
# -----------------------------------

project = Path(__file__).resolve().parent
results = project / "results"
results.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device, flush=True)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

# -----------------------------------
# 3. Load training and validation data
# -----------------------------------

train_df = pd.read_csv(results / "train_split.csv")
val_df = pd.read_csv(results / "validation_split.csv")

train_dataset = FacialFeatureDataset(
    train_df,
    project / "train",
    train_transform
)

val_dataset = FacialFeatureDataset(
    val_df,
    project / "train",
    val_transform
)

BATCH_SIZE = 16
EPOCHS = 5

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# -----------------------------------
# 4. Build ResNet18
# -----------------------------------

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze pretrained feature extraction layers
for parameter in model.parameters():
    parameter.requires_grad = False

# New final layer for 8 competition classes
model.fc = nn.Linear(
    model.fc.in_features,
    len(class_to_idx)
)

model = model.to(device)

# -----------------------------------
# 5. Training settings
# -----------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

best_accuracy = -1.0
history = []

# Save class mapping for Kaggle predictions
with open(results / "class_to_idx.json", "w") as f:
    json.dump(class_to_idx, f, indent=2)

print("Training images:", len(train_dataset), flush=True)
print("Validation images:", len(val_dataset), flush=True)
print("Batch size:", BATCH_SIZE, flush=True)
print("Epochs:", EPOCHS, flush=True)
print("Starting training...", flush=True)

# -----------------------------------
# 6. Train and validate
# -----------------------------------

for epoch in range(EPOCHS):

    start = time.time()

    model.train()

    # Keep frozen BatchNorm statistics fixed
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_number, (images, labels) in enumerate(
        train_loader, start=1
    ):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += labels.size(0)

        # Show progress every 200 batches
        if batch_number % 200 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Training batch {batch_number}/{len(train_loader)}",
                flush=True
            )

    # Validation
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_total += labels.size(0)

    train_accuracy = train_correct / train_total
    val_accuracy = val_correct / val_total

    epoch_results = {
        "epoch": epoch + 1,
        "train_loss": train_loss / train_total,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss / val_total,
        "val_accuracy": val_accuracy
    }

    history.append(epoch_results)

    elapsed = (time.time() - start) / 60

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} completed | "
        f"Train accuracy: {train_accuracy:.2%} | "
        f"Validation accuracy: {val_accuracy:.2%} | "
        f"Validation loss: {epoch_results['val_loss']:.4f} | "
        f"Time: {elapsed:.1f} minutes\n",
        flush=True
    )

    # Save the best checkpoint
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            project / "best_resnet18.pth"
        )

        print("New best model saved!", flush=True)

    # Record experiment results
    pd.DataFrame(history).to_csv(
        results / "training_history.csv",
        index=False
    )

print("\nTraining completed!", flush=True)
print(
    f"Best validation accuracy: {best_accuracy:.2%}",
    flush=True
)

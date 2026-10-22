
# ENGE707 Assessment 3 - Task 6
# Fine-tune ResNet18 to improve Kaggle classification accuracy

from pathlib import Path
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

# ----------------------------------------
# 1. Setup
# ----------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

project = Path(__file__).resolve().parent
results = project / "results"
results.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device, flush=True)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

# ----------------------------------------
# 2. Load dataset
# ----------------------------------------

train_df = pd.read_csv(results / "train_split.csv")
val_df = pd.read_csv(results / "validation_split.csv")

train_dataset = FacialFeatureDataset(
    train_df,
    project / "train",
    train_transform,
)

val_dataset = FacialFeatureDataset(
    val_df,
    project / "train",
    val_transform,
)

BATCH_SIZE = 16
EPOCHS = 6

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

# ----------------------------------------
# 3. Load best baseline model
# ----------------------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_to_idx),
)

model.load_state_dict(
    torch.load(
        project / "best_resnet18.pth",
        map_location="cpu",
        weights_only=True,
    )
)

# Freeze everything initially
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze final convolutional block
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Keep classification layer trainable
for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(device)

# ----------------------------------------
# 4. Optimiser and loss
# ----------------------------------------

criterion = nn.CrossEntropyLoss()

# Smaller learning rate for pretrained features
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.layer4.parameters(),
            "lr": 0.00002,
        },
        {
            "params": model.fc.parameters(),
            "lr": 0.0001,
        },
    ],
    weight_decay=0.0001,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

# Our baseline achieved 90.53%.
# Start at -1 so the first fine-tuned model
# is saved even if it performs worse.
best_accuracy = -1.0
history = []

checkpoint_path = project / "best_resnet18_finetuned.pth"

print("Training images:", len(train_dataset), flush=True)
print("Validation images:", len(val_dataset), flush=True)
print("Fine-tuning epochs:", EPOCHS, flush=True)
print("Starting fine-tuning...", flush=True)

# ----------------------------------------
# 5. Fine-tune model
# ----------------------------------------

for epoch in range(EPOCHS):

    start = time.time()

    model.train()

    # Keep BatchNorm running statistics fixed
    # while allowing layer4 weights to learn.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_number, (images, labels) in enumerate(
        train_loader, start=1
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

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

        if batch_number % 200 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Training batch {batch_number}/{len(train_loader)}",
                flush=True,
            )

    # ------------------------------------
    # Validation
    # ------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)

            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            val_total += labels.size(0)

    train_accuracy = train_correct / train_total
    val_accuracy = val_correct / val_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss / train_total,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss / val_total,
        "val_accuracy": val_accuracy,
    })

    elapsed = (time.time() - start) / 60

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} completed | "
        f"Train accuracy: {train_accuracy:.2%} | "
        f"Validation accuracy: {val_accuracy:.2%} | "
        f"Validation loss: {val_loss / val_total:.4f} | "
        f"Time: {elapsed:.1f} minutes\n",
        flush=True,
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

        print(
            "New best fine-tuned model saved!",
            flush=True,
        )

    pd.DataFrame(history).to_csv(
        results / "finetuning_history.csv",
        index=False,
    )

    scheduler.step()

print("\nFine-tuning completed!", flush=True)

print(
    f"Best fine-tuned validation accuracy: {best_accuracy:.2%}",
    flush=True,
)

print(
    "Baseline validation accuracy: 90.53%",
    flush=True,
)

print(
    "Saved model:",
    checkpoint_path,
    flush=True,
)

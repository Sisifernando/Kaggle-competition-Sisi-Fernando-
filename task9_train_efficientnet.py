
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

# ============================================================
# TASK 9A - EFFICIENTNET-B0 TRAINING AND FINE-TUNING
# ============================================================

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


# ============================================================
# 1. LOAD TRAINING AND VALIDATION DATA
# ============================================================

train_df = pd.read_csv(
    results / "train_split.csv"
)

val_df = pd.read_csv(
    results / "validation_split.csv"
)

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
EPOCHS = 8

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

print("Training images:", len(train_dataset), flush=True)
print("Validation images:", len(val_dataset), flush=True)
print("Batch size:", BATCH_SIZE, flush=True)
print("Training epochs:", EPOCHS, flush=True)


# ============================================================
# 2. LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print("\nLoading pretrained EfficientNet-B0...", flush=True)

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

# Replace the original classifier with 8 output classes.
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_to_idx),
)

# Freeze all model parameters first.
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final EfficientNet feature blocks.
for parameter in model.features[6:].parameters():
    parameter.requires_grad = True

# Unfreeze the classifier.
for parameter in model.classifier.parameters():
    parameter.requires_grad = True

model = model.to(device)

print("EfficientNet-B0 loaded successfully!", flush=True)


# ============================================================
# 3. TRAINING CONFIGURATION
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.features[6:].parameters(),
            "lr": 0.00002,
        },
        {
            "params": model.classifier.parameters(),
            "lr": 0.0001,
        },
    ],
    weight_decay=0.0001,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

best_accuracy = -1.0
history = []

checkpoint_path = (
    project / "best_efficientnet_b0.pth"
)

print("\nStarting EfficientNet-B0 training...", flush=True)


# ============================================================
# 4. TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    start_time = time.time()

    model.train()

    # Freeze BatchNorm running statistics.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_number, (images, labels) in enumerate(
        train_loader,
        start=1,
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

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        train_loss += (
            loss.item() * labels.size(0)
        )

        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        train_total += labels.size(0)

        if batch_number % 200 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Training batch "
                f"{batch_number}/{len(train_loader)}",
                flush=True,
            )


    # ========================================================
    # 5. VALIDATION LOOP
    # ========================================================

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

            loss = criterion(
                outputs,
                labels,
            )

            val_loss += (
                loss.item() * labels.size(0)
            )

            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            val_total += labels.size(0)


    # ========================================================
    # 6. CALCULATE EPOCH RESULTS
    # ========================================================

    train_accuracy = (
        train_correct / train_total
    )

    val_accuracy = (
        val_correct / val_total
    )

    average_train_loss = (
        train_loss / train_total
    )

    average_val_loss = (
        val_loss / val_total
    )

    elapsed_minutes = (
        time.time() - start_time
    ) / 60

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": average_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": average_val_loss,
            "val_accuracy": val_accuracy,
        }
    )

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} completed\n"
        f"Training accuracy: {train_accuracy:.2%}\n"
        f"Validation accuracy: {val_accuracy:.2%}\n"
        f"Validation loss: {average_val_loss:.4f}\n"
        f"Time: {elapsed_minutes:.1f} minutes\n",
        flush=True,
    )


    # ========================================================
    # 7. SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

        print(
            "New best EfficientNet model saved!",
            flush=True,
        )


    # Save training history after each epoch.
    pd.DataFrame(history).to_csv(
        results / "efficientnet_training_history.csv",
        index=False,
    )

    scheduler.step()


# ============================================================
# 8. FINAL RESULTS
# ============================================================

print("\n==================================")
print("TASK 9A COMPLETED SUCCESSFULLY")
print("==================================")

print(
    f"Best EfficientNet validation accuracy: "
    f"{best_accuracy:.2%}"
)

print(
    "Previous ResNet18 + TTA validation accuracy: 98.88%"
)

print(
    "Model saved to:",
    checkpoint_path,
)

print(
    "Training history saved to:",
    results / "efficientnet_training_history.csv",
)


from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF
from PIL import Image

from task2_image_pipeline import (
    FacialFeatureDataset,
    val_transform,
    class_to_idx,
)

# ==========================================================
# TASK 9B - RESNET18 + EFFICIENTNET-B0 ENSEMBLE WITH TTA
# ==========================================================

project = Path(__file__).resolve().parent
results = project / "results"
results.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16

# We will test these weights on the validation dataset.
# The first number is ResNet18, second is EfficientNet.
WEIGHTS_TO_TEST = [
    (0.50, 0.50),
    (0.30, 0.70),
]

print("Device:", device, flush=True)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

# ==========================================================
# 1. LOAD BOTH TRAINED MODELS
# ==========================================================

resnet = models.resnet18(weights=None)

resnet.fc = nn.Linear(
    resnet.fc.in_features,
    len(class_to_idx),
)

resnet.load_state_dict(
    torch.load(
        project / "best_resnet18_finetuned.pth",
        map_location="cpu",
        weights_only=True,
    )
)

resnet = resnet.to(device)
resnet.eval()

efficientnet = models.efficientnet_b0(weights=None)

efficientnet.classifier[1] = nn.Linear(
    efficientnet.classifier[1].in_features,
    len(class_to_idx),
)

efficientnet.load_state_dict(
    torch.load(
        project / "best_efficientnet_b0.pth",
        map_location="cpu",
        weights_only=True,
    )
)

efficientnet = efficientnet.to(device)
efficientnet.eval()

print("Both trained models loaded successfully!", flush=True)


# ==========================================================
# 2. FUNCTION TO GET TTA PROBABILITIES
# ==========================================================

def predict_tta(model, images):
    original_probs = torch.softmax(
        model(images), dim=1
    )

    flipped_probs = torch.softmax(
        model(TF.hflip(images)), dim=1
    )

    return (original_probs + flipped_probs) / 2


# ==========================================================
# 3. VALIDATION: COMPARE ENSEMBLE WEIGHTS
# ==========================================================

val_df = pd.read_csv(
    results / "validation_split.csv"
)

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

correct = {
    weights: 0 for weights in WEIGHTS_TO_TEST
}

resnet_correct = 0
efficientnet_correct = 0
total = 0

print("\nEvaluating ensembles on validation data...", flush=True)

with torch.inference_mode():

    for batch_number, (images, labels) in enumerate(
        val_loader, start=1
    ):

        images = images.to(device)
        labels = labels.to(device)

        resnet_probs = predict_tta(resnet, images)
        efficientnet_probs = predict_tta(
            efficientnet, images
        )

        resnet_correct += (
            resnet_probs.argmax(dim=1) == labels
        ).sum().item()

        efficientnet_correct += (
            efficientnet_probs.argmax(dim=1) == labels
        ).sum().item()

        for weights in WEIGHTS_TO_TEST:
            resnet_weight, efficientnet_weight = weights

            combined_probs = (
                resnet_weight * resnet_probs
                + efficientnet_weight * efficientnet_probs
            )

            correct[weights] += (
                combined_probs.argmax(dim=1) == labels
            ).sum().item()

        total += labels.size(0)

        if batch_number % 200 == 0:
            print(
                f"Validation batch {batch_number}/{len(val_loader)}",
                flush=True,
            )


# ==========================================================
# 4. SAVE COMPARISON AND SELECT BEST WEIGHTS
# ==========================================================

rows = [
    {
        "method": "ResNet18 + TTA",
        "validation_accuracy": resnet_correct / total,
    },
    {
        "method": "EfficientNet-B0 + TTA",
        "validation_accuracy": efficientnet_correct / total,
    },
]

for weights in WEIGHTS_TO_TEST:
    rows.append(
        {
            "method": f"Ensemble {weights[0]:.0%} ResNet / {weights[1]:.0%} EfficientNet",
            "validation_accuracy": correct[weights] / total,
        }
    )

comparison = pd.DataFrame(rows)

comparison.to_csv(
    results / "ensemble_validation_comparison.csv",
    index=False,
)

print("\nVALIDATION RESULTS")
for row in rows:
    print(
        f"{row['method']}: "
        f"{row['validation_accuracy']:.2%}"
    )

best_weights = max(
    WEIGHTS_TO_TEST,
    key=lambda weights: correct[weights],
)

best_accuracy = correct[best_weights] / total

print(
    f"\nBest ensemble: {best_weights[0]:.0%} ResNet / "
    f"{best_weights[1]:.0%} EfficientNet"
)
print(f"Best ensemble accuracy: {best_accuracy:.2%}")

best_single_accuracy = max(
    resnet_correct / total,
    efficientnet_correct / total,
)

if best_accuracy <= best_single_accuracy:
    print(
        "\nNOTE: Ensemble did not beat the best single model "
        "on validation. Consider keeping the existing "
        "Kaggle submission as your final model."
    )

# ==========================================================
# 5. LOAD KAGGLE TEST IMAGES
# ==========================================================

sample = pd.read_csv(
    project / "sample_submission.csv"
)

assert {"image_id", "class"}.issubset(sample.columns)

test_files = {
    path.stem: path
    for path in (project / "test").rglob("*")
    if path.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }
}


class TestDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_id = str(
            self.dataframe.iloc[index]["image_id"]
        )

        image_path = test_files.get(image_id)

        if image_path is None:
            image_path = test_files.get(
                image_id.zfill(5)
            )

        if image_path is None:
            raise FileNotFoundError(
                f"Missing test image: {image_id}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return val_transform(image)


test_dataset = TestDataset(sample)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

print("\nTest images:", len(test_dataset), flush=True)

# ==========================================================
# 6. GENERATE ENSEMBLE KAGGLE PREDICTIONS
# ==========================================================

predictions = []

resnet_weight, efficientnet_weight = best_weights

idx_to_class = {
    index: name
    for name, index in class_to_idx.items()
}

print("Generating ensemble predictions...", flush=True)

with torch.inference_mode():

    for batch_number, images in enumerate(
        test_loader, start=1
    ):

        images = images.to(device)

        resnet_probs = predict_tta(resnet, images)
        efficientnet_probs = predict_tta(
            efficientnet, images
        )

        combined_probs = (
            resnet_weight * resnet_probs
            + efficientnet_weight * efficientnet_probs
        )

        predicted_indices = (
            combined_probs.argmax(dim=1)
            .cpu()
            .tolist()
        )

        predictions.extend(
            idx_to_class[index]
            for index in predicted_indices
        )

        if batch_number % 200 == 0:
            print(
                f"Test batch {batch_number}/{len(test_loader)}",
                flush=True,
            )


# ==========================================================
# 7. SAVE FINAL SUBMISSION
# ==========================================================

assert len(predictions) == len(sample), (
    "Prediction count mismatch!"
)

submission = sample.copy()
submission["class"] = predictions

output_path = (
    results / "submission_ensemble_tta.csv"
)

submission.to_csv(
    output_path,
    index=False,
)

print("\nTASK 9B COMPLETED SUCCESSFULLY!")
print("Saved to:", output_path)
print("Total predictions:", len(submission))
print(
    f"Selected weights: "
    f"{resnet_weight:.0%} ResNet / "
    f"{efficientnet_weight:.0%} EfficientNet"
)
print(submission.head())

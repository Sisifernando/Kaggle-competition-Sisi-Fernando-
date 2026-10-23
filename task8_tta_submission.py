
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF
from PIL import Image

from task2_image_pipeline import val_transform, class_to_idx

# Task 8C: Kaggle submission using horizontal-flip TTA.

project = Path(__file__).resolve().parent
results = project / "results"
results.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

print("Device:", device, flush=True)

sample = pd.read_csv(project / "sample_submission.csv")

print("Submission columns:", sample.columns.tolist())
print("Number of test images:", len(sample))

# Locate test images using their filename stems.
test_files = {
    path.stem: path
    for path in (project / "test").rglob("*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
}

print("Test image files found:", len(test_files))


class TestDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_id = str(self.dataframe.iloc[index]["image_id"])
        image_path = test_files.get(image_id.zfill(5))

        if image_path is None:
            raise FileNotFoundError(f"Missing test image: {image_id}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = val_transform(image)

        return image


test_dataset = TestDataset(sample)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

# Load the best fine-tuned ResNet18 model.
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

idx_to_class = {
    index: class_name
    for class_name, index in class_to_idx.items()
}

predictions = []

print("Generating TTA predictions...", flush=True)

with torch.inference_mode():
    for batch_number, images in enumerate(test_loader, start=1):
        images = images.to(device)

        # Predict original images.
        original_logits = model(images)
        original_probs = torch.softmax(original_logits, dim=1)

        # Predict horizontally flipped images.
        flipped_images = TF.hflip(images)
        flipped_logits = model(flipped_images)
        flipped_probs = torch.softmax(flipped_logits, dim=1)

        # Average probabilities from both predictions.
        average_probs = (original_probs + flipped_probs) / 2

        predicted_indices = average_probs.argmax(dim=1).cpu().tolist()

        predictions.extend(
            idx_to_class[index] for index in predicted_indices
        )

        if batch_number % 200 == 0:
            print(
                f"Processed {batch_number}/{len(test_loader)} batches",
                flush=True,
            )

assert len(predictions) == len(sample), "Prediction count mismatch!"

submission = sample.copy()
submission["class"] = predictions

output_path = results / "submission_resnet18_tta.csv"
submission.to_csv(output_path, index=False)

print("\nTask 8C completed successfully!")
print("Saved to:", output_path)
print("Total predictions:", len(submission))
print("\nFirst five predictions:")
print(submission.head())


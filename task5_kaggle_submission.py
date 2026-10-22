
# ENGE707 Assessment 3 - Task 5
# Generate Kaggle competition predictions using the best ResNet18

from pathlib import Path
import json

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

project = Path(__file__).resolve().parent
results = project / "results"
results.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device, flush=True)

# -----------------------------------
# 1. Read Kaggle submission template
# -----------------------------------

sample = pd.read_csv(project / "sample_submission.csv")

print("Submission columns:", sample.columns.tolist())
print("Number of test images:", len(sample))

# Expected: image_id and class
assert "image_id" in sample.columns, (
    "Could not find image_id in sample_submission.csv"
)
assert "class" in sample.columns, (
    "Could not find class in sample_submission.csv"
)

# -----------------------------------
# 2. Load class mapping
# -----------------------------------

with open(results / "class_to_idx.json", "r") as file:
    class_to_idx = json.load(file)

idx_to_class = {
    index: name
    for name, index in class_to_idx.items()
}

# -----------------------------------
# 3. Match validation preprocessing
# -----------------------------------

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------
# 4. Create test dataset
# -----------------------------------

class KaggleTestDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform):
        self.dataframe = dataframe
        self.transform = transform

        self.image_paths = {
            path.stem: path
            for path in image_dir.rglob("*")
            if path.suffix.lower() in [
                ".jpg", ".jpeg", ".png", ".webp"
            ]
        }

        print(
            "Test image files found:",
            len(self.image_paths),
            flush=True
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_id = str(
            self.dataframe.iloc[index]["image_id"]
        )

        image_path = self.image_paths.get(image_id)

        if image_path is None:
            image_path = self.image_paths.get(
                image_id.zfill(5)
            )

        if image_path is None:
            raise FileNotFoundError(
                f"Could not locate test image {image_id}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image

test_dataset = KaggleTestDataset(
    sample,
    project / "test",
    test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# -----------------------------------
# 5. Load best trained ResNet18
# -----------------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_to_idx)
)

model.load_state_dict(
    torch.load(
        project / "best_resnet18.pth",
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

# -----------------------------------
# 6. Predict Kaggle test images
# -----------------------------------

predicted_classes = []

print("Generating Kaggle predictions...", flush=True)

with torch.inference_mode():

    for batch_number, images in enumerate(
        test_loader, start=1
    ):

        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        predicted_classes.extend([
            idx_to_class[index]
            for index in predictions.cpu().tolist()
        ])

        if batch_number % 200 == 0:
            print(
                f"Processed {batch_number}/{len(test_loader)} batches",
                flush=True
            )

# -----------------------------------
# 7. Save Kaggle submission CSV
# -----------------------------------

assert len(predicted_classes) == len(sample)

submission = sample.copy()
submission["class"] = predicted_classes

output_path = results / "submission_resnet18.csv"

submission.to_csv(output_path, index=False)

print("\nSubmission created successfully!")
print("Saved to:", output_path)
print("Total predictions:", len(submission))
print("\nFirst five rows:")
print(submission.head())


# ENGE707 Assessment 3 - Task 2
# Image preprocessing and data augmentation

from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

project = Path(__file__).resolve().parent

train_df = pd.read_csv(project / "results/train_split.csv")
val_df = pd.read_csv(project / "results/validation_split.csv")

# Define eight classes consistently
classes = sorted(train_df["class"].unique())
class_to_idx = {name: i for i, name in enumerate(classes)}

print("Class encoding:", class_to_idx)

# Training augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation: no random augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class FacialFeatureDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        self.files = {}
        for path in image_dir.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                self.files[path.stem] = path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_id = str(row["image_id"])

        path = self.files.get(image_id)
        if path is None:
            path = self.files.get(image_id.zfill(5))

        if path is None:
            raise FileNotFoundError(f"Missing image: {image_id}")

        with Image.open(path) as img:
            image = img.convert("RGB")
            image = self.transform(image)

        label = class_to_idx[row["class"]]
        return image, label

train_dataset = FacialFeatureDataset(
    train_df, project / "train", train_transform
)

val_dataset = FacialFeatureDataset(
    val_df, project / "train", val_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=0
)


# Only run this test when executing this file directly
if __name__ == "__main__":
    images, labels = next(iter(train_loader))

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Image pipeline working successfully!")


import pandas as pd
from pathlib import Path

# TASK 1: DATASET EXPLORATION
# ENGE707 Assessment 3

# Find project folder
project = Path(__file__).resolve().parent

# Load training labels
labels = pd.read_csv(project / "train_labels.csv")

print("=== ENGE707 DATASET EXPLORATION ===")

# Basic information
print("\nNumber of labelled images:", len(labels))
print("\nColumn names:")
print(labels.columns.tolist())

print("\nFirst five rows:")
print(labels.head())

# Check missing values
print("\nMissing values:")
print(labels.isna().sum())

# Check duplicates
print("\nDuplicate rows:", labels.duplicated().sum())

# Count training and test images
extensions = {".jpg", ".jpeg", ".png", ".webp"}

train_folder = project / "train"
test_folder = project / "test"

train_images = [
    p for p in train_folder.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

test_images = [
    p for p in test_folder.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

print("\nTraining images found:", len(train_images))
print("Test images found:", len(test_images))

# Count images per class
if "class" in labels.columns:
    print("\nImages per class:")
    print(labels["class"].value_counts())

    print("\nClass percentages:")
    print(
        (labels["class"].value_counts(normalize=True) * 100)
        .round(2)
    )
else:
    print("\nClass column not found.")
    print("Check the column names printed above.")

print("\n=== EXPLORATION COMPLETE ===")

# Count the images in each class
print("\n=== CLASS DISTRIBUTION ===")

class_counts = labels["class"].value_counts()

print(class_counts)

print("\nClass percentages:")
print((class_counts / len(labels) * 100).round(2))

print("\nNumber of classes:", labels["class"].nunique())

# TASK 1 - Class distribution graph

import matplotlib.pyplot as plt

# Count the images in each class
counts = labels["class"].value_counts().sort_index()

# Create graph
plt.figure(figsize=(11, 6))
bars = plt.bar(counts.index, counts.values)

plt.title("ENGE707 Training Dataset - Class Distribution")
plt.xlabel("Image Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)

# Display numbers above bars
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 100,
        str(int(bar.get_height())),
        ha="center"
    )

plt.tight_layout()

# Save graph for report
plt.savefig("class_distribution.png", dpi=300)

# Show graph
plt.show()

print("Class distribution graph saved!")

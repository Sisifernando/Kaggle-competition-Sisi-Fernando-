
# ENGE707 Assessment 3 - Task 2
# Stratified training and validation split

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

project = Path(__file__).resolve().parent

# Load labelled data
labels = pd.read_csv(project / "train_labels.csv")

# Keep the same split for reproducibility
train_df, val_df = train_test_split(
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels["class"]
)

# Save split information
output = project / "results"
output.mkdir(exist_ok=True)

train_df.to_csv(output / "train_split.csv", index=False)
val_df.to_csv(output / "validation_split.csv", index=False)

print("=== TASK 2: DATA PREPARATION ===")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))

print("\nTraining class distribution:")
print(train_df["class"].value_counts())

print("\nValidation class distribution:")
print(val_df["class"].value_counts())

print("\nTraining percentages:")
print(
    (train_df["class"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nValidation percentages:")
print(
    (val_df["class"].value_counts(normalize=True) * 100)
    .round(2)
)

# Check that images are not shared across splits
overlap = set(train_df["image_id"]) & set(val_df["image_id"])

print("\nOverlapping image IDs:", len(overlap))
assert len(overlap) == 0, "Data leakage detected!"

print("\nTask 2 split completed successfully!")


from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

project = Path(__file__).resolve().parent
labels = pd.read_csv(project / "train_labels.csv")

train_folder = project / "train"
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

# Find training images
images = {
    p.stem: p
    for p in train_folder.rglob("*")
    if p.is_file() and p.suffix.lower() in image_extensions
}

# Match image IDs with their labels
def find_image(image_id):
    name = str(image_id)
    return images.get(name) or images.get(name.zfill(5))

print("=== IMAGE QUALITY INSPECTION ===")

sizes = Counter()
formats = Counter()
corrupted = []
missing = []

# Check image dimensions and readability
for image_id in labels["image_id"]:
    path = find_image(image_id)

    if path is None:
        missing.append(image_id)
        continue

    try:
        with Image.open(path) as img:
            sizes[img.size] += 1
            formats[img.format] += 1
            img.verify()
    except (OSError, ValueError, UnidentifiedImageError):
        corrupted.append(str(path))

print("\nMost common image dimensions:")
for size, count in sizes.most_common(10):
    print(size, count)

print("\nImage formats:")
print(formats)

print("\nMissing image files:", len(missing))
print("Corrupted images:", len(corrupted))

# Display one representative image per class
classes = sorted(labels["class"].unique())

fig, axes = plt.subplots(2, 4, figsize=(13, 7))

for ax, class_name in zip(axes.flat, classes):
    row = labels[labels["class"] == class_name].iloc[0]
    path = find_image(row["image_id"])

    if path is not None:
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(class_name)
    else:
        ax.set_title(f"{class_name} - missing")

    ax.axis("off")

plt.suptitle("Representative Images from the Eight Classes")
plt.tight_layout()
plt.savefig(project / "sample_images.png", dpi=200)
plt.show()

print("\nSaved sample_images.png")
print("=== INSPECTION COMPLETE ===")

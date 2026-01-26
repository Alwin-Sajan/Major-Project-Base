import os
import shutil
import random

def split_dataset(
    source_dir,
    output_dir,
    train_ratio=0.8,
    seed=42
):
    random.seed(seed)

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Loop through class folders
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        # Skip non-folder files
        if not os.path.isdir(class_path):
            continue

        # Make class folders in train & val
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # List all images
        images = os.listdir(class_path)
        random.shuffle(images)

        # Split
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Copy files
        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in val_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)

        print(f"[✔] {class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

    print("\nDataset split completed!")


# ------- Usage --------

source_dataset = "/media/abk/New Disk/DATASETS/first/archive"          # Your original folder
output_dataset = "/media/abk/New Disk/DATASETS/first/updatedDataset"    # New location for train/val

split_dataset(source_dataset, output_dataset, train_ratio=0.90)

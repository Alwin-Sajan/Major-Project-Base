import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --------------------------------------------------
# IMPORT MODEL DEFINITIONS (FROM TRAINING FILE)
# --------------------------------------------------
from train_convvnext import ConvNeXtIncremental, CosineClassifier
# ↑ filename must match exactly

# --------------------------------------------------
# PATHS
# --------------------------------------------------
val_dir = "/media/abk/New Disk/DATASETS/first/updatedDataset/val"
checkpoint_path = "models/convnext_best_weights.pth"

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
batch_size = 32
embedding_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# TRANSFORMS (MUST MATCH TRAINING)
# --------------------------------------------------
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# DATASET
# --------------------------------------------------
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
num_classes = len(val_dataset.classes)

# --------------------------------------------------
# BUILD MODELS (EXACT MATCH)
# --------------------------------------------------
model = ConvNeXtIncremental(
    embedding_dim=embedding_dim,
    pretrained=False  # IMPORTANT: do NOT load ImageNet again
)

classifier = CosineClassifier(
    embedding_dim=embedding_dim,
    n_classes=num_classes
)

# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------
checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint["model_state"])
classifier.load_state_dict(checkpoint["classifier_state"])

model.to(device).eval()
classifier.to(device).eval()

# --------------------------------------------------
# ACCURACY COMPUTATION
# --------------------------------------------------
correct = 0
total = 0
class_correct = [0] * num_classes
class_total = [0] * num_classes

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)               # ConvNeXt → embedding
        logits = classifier(embeddings)          # cosine logits
        preds = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += (preds[i] == label).item()
            class_total[label] += 1

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
print(f"\nOverall Accuracy: {100 * correct / total:.2f}%")

print("\nPer-class Accuracy:")
for i, cls in enumerate(val_dataset.classes):
    acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"{cls}: {acc:.2f}%")

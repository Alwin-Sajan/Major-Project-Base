import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_convvnext import ConvNeXtIncremental, CosineClassifier

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# PATHS
# --------------------------------------------------
val_dir = "/media/abk/New Disk/DATASETS/first/updatedDataset/val"
checkpoint_path = "/home/abk/abk/projects/Major-project-basic-ui/models/convnext_best_weights.pth"

batch_size = 32
embedding_dim = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
num_classes = len(val_dataset.classes)

# --------------------------------------------------
# BUILD MODELS
# --------------------------------------------------
model = ConvNeXtIncremental(
    embedding_dim=embedding_dim,
    pretrained=False
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
# ACCURACY + STORE PREDICTIONS
# --------------------------------------------------
correct = 0
total = 0
class_correct = [0] * num_classes
class_total = [0] * num_classes

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        logits = classifier(embeddings)
        preds = logits.argmax(dim=1)

        # Store predictions for confusion matrix
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += (preds[i] == label).item()
            class_total[label] += 1

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------
print(f"\nOverall Accuracy: {100 * correct / total:.2f}%")

print("\nPer-class Accuracy:")
for i, cls in enumerate(val_dataset.classes):
    acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"{cls}: {acc:.2f}%")

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=val_dataset.classes,
    yticklabels=val_dataset.classes
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

# Save image
# plt.savefig(r"/home/abk/abk/projects/Major-project-basic-ui/backend/Results/CONVNEXT_confusion_matrix.png", dpi=300)

# Display image
plt.show()
plt.close()

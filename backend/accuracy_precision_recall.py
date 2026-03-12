import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_convvnext import ConvNeXtIncremental, CosineClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# --------------------------------------------------
# PATHS & CONFIG
# --------------------------------------------------
val_dir = "/media/abk/New Disk/DATASETS/first/updatedDataset/val"
checkpoint_path = "/home/abk/abk/projects/Major-project-basic-ui/models/convnext_best_weights.pth"
batch_size = 32
embedding_dim = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# TRANSFORMS & DATALOADER
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
# BUILD & LOAD MODELS
# --------------------------------------------------
model = ConvNeXtIncremental(
    embedding_dim=embedding_dim,
    pretrained=False
)
classifier = CosineClassifier(
    embedding_dim=embedding_dim,
    n_classes=num_classes
)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
classifier.load_state_dict(checkpoint["classifier_state"])

model.to(device).eval()
classifier.to(device).eval()

# --------------------------------------------------
# INFERENCE LOOP
# --------------------------------------------------
all_preds = []
all_labels = []

print("Running inference on validation set...")
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        embeddings = model(images)
        logits = classifier(embeddings)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
y_true = np.array(all_labels)
y_pred = np.array(all_preds)

# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------
# Macro: Calculates metrics for each label, and finds their unweighted mean. 
# This does not take label imbalance into account.
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
overall_accuracy = accuracy_score(y_true, y_pred)

# Per-class metrics
prec_per_cls, rec_per_cls, f1_per_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

# --------------------------------------------------
# PRINT DETAILED RESULTS
# --------------------------------------------------
print("\n" + "="*50)
print("FINAL EVALUATION RESULTS")
print("="*50)
print(f"{'Overall Accuracy:':<20} {overall_accuracy * 100:.2f}%")
print(f"{'Precision:':<20} {precision_macro * 100:.2f}%")
print(f"{'Recall:':<20} {recall_macro * 100:.2f}%")
print(f"{'F1-Score:':<20} {f1_macro * 100:.2f}%")
print("-" * 50)

print("\nPER-CLASS METRICS:")
header = f"{'Class Name':<20} | {'Prec. (%)':<10} | {'Rec. (%)':<10} | {'F1 (%)':<10}"
print(header)
print("-" * len(header))

for i, class_name in enumerate(val_dataset.classes):
    print(f"{class_name:<20} | {prec_per_cls[i]*100:>9.2f} | {rec_per_cls[i]*100:>9.2f} | {f1_per_cls[i]*100:>9.2f}")

# Alternative: Scikit-learn's built-in formatted report
print("\n" + "="*50)
print("SKLEARN CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred, target_names=val_dataset.classes))
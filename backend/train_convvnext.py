import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


# =====================================================
# 1. Metric-Learning Backbone
# =====================================================
class ConvNeXtIncremental(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.convnext_tiny(weights=weights)
        
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        in_features = backbone.classifier[2].in_features 
        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        emb = self.proj(x)
        return F.normalize(emb, p=2, dim=1)


# =====================================================
# 2. Distance-Based Classifier
# =====================================================
class CosineClassifier(nn.Module):
    def __init__(self, embedding_dim, n_classes):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_classes, embedding_dim))
        self.scale = nn.Parameter(torch.tensor(20.0)) 


    def forward(self, emb):
        normalized_centers = F.normalize(self.centers, p=2, dim=1)
        logits = torch.mm(emb, normalized_centers.t())
        return logits * self.scale


# =====================================================
# 3. Validation & Prototype Utils
# =====================================================
def evaluate(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = classifier(model(x))
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / total


@torch.no_grad()
def build_final_prototypes(model, loader, device):
    model.eval()
    all_embs, all_labs = [], []
    for x, y in loader:
        all_embs.append(model(x.to(device)).cpu())
        all_labs.append(y)
    
    embs = torch.cat(all_embs)
    labs = torch.cat(all_labs)
    unique_labels = torch.unique(labs)
    
    prototypes = torch.zeros(len(unique_labels), embs.shape[1])
    for label in unique_labels:
        prototypes[label] = embs[labs == label].mean(0)
    
    return F.normalize(prototypes, p=2, dim=1)


# =====================================================
# 4. Training Loop with Best-Model Saving
# =====================================================
def train_model(data_root, epochs=40, batch_size=32, lr=1e-4, device="cuda"):
    # Setup Data
    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # optional but good
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


    model = ConvNeXtIncremental().to(device)
    classifier = CosineClassifier(256, len(train_ds.classes)).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 3e-5},
        {'params': model.proj.parameters(),     'lr': 1e-4},
        {'params': classifier.parameters(),     'lr': 1e-4},
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0.0
    os.makedirs("models", exist_ok=True)


    patience = 5 
    epochs_no_improve = 0
    min_delta = 1e-4    


    for epoch in range(epochs):
        model.train()
        classifier.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            logits = classifier(model(x))
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})


        val_acc = evaluate(model, classifier, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")


        # --- SAVE BEST MODEL AFTER EACH EPOCH ---
        prototypes = build_final_prototypes(model, train_loader, device)

        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'classifier_state': classifier.state_dict(),
                'prototypes': prototypes,
                'classes': train_ds.classes,
                'acc': val_acc
            }, "models/convnext_best_weights.pth")
            print(f"⭐ New Best Model Saved (Acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")


        if epochs_no_improve >= patience:
            print("🛑 Early stopping triggered")
            break


    # --- FINAL STEP: GENERATE PROTOTYPES FOR OOD ---
    print("Training finished. Loading best weights to generate final prototypes...")
    checkpoint = torch.load("models/convnext_best_weights.pth")
    model.load_state_dict(checkpoint['model_state'])
    
    prototypes = build_final_prototypes(model, train_loader, device)

    classifier = CosineClassifier(256, len(datasets.ImageFolder(os.path.join(data_root, "train"), train_transform).classes)).to(device)
    # torch.save({
    #     'model_state': model.state_dict(),
    #     'prototypes': prototypes,
    #     'classes': train_ds.classes,
    #     'classifier_state': classifier.state_dict(),
    # }, "models/convnext_final_for_ood.pth")
    print("✅ Final Model with Prototypes Saved.")


if __name__ == "__main__":
    DATA_PATH = "/media/abk/New Disk/DATASETS/first/updatedDataset"
    train_model(DATA_PATH, epochs=5)
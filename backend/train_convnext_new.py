import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

EMBEDDING_DIM = 512

# =====================================================
# 1. Metric-Learning Backbone
# =====================================================
class ConvNeXtIncremental(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True, dropout=0.3):
        super().__init__()
        # Use Weights enum for modern torchvision
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.convnext_tiny(weights=weights)
        
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        in_features = backbone.classifier[2].in_features 
        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
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
        nn.init.kaiming_uniform_(self.centers, a=np.sqrt(5))

    def forward(self, emb):
        normalized_centers = F.normalize(self.centers, p=2, dim=1)
        logits = torch.mm(emb, normalized_centers.t())
        return logits * self.scale

# =====================================================
# 3. Utilities
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
    for i, label in enumerate(unique_labels):
        prototypes[i] = embs[labs == label].mean(0)
    
    return F.normalize(prototypes, p=2, dim=1)

# =====================================================
# 4. Training Loop (Fixed Logic)
# =====================================================
def train_model(data_root, epochs=50, batch_size=32, device="cuda"):
    # FIX: Uniform transforms for both Training and Inference
    # Prevents "Squashing" images which hurts Whale/Fish detection
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224), # Safer than aggressive RandomResizedCrop
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), # Reduced intensity
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ConvNeXtIncremental(embedding_dim=EMBEDDING_DIM).to(device)
    classifier = CosineClassifier(EMBEDDING_DIM, len(train_ds.classes)).to(device)
    
    # FIX: Differential Learning Rates (Backbone learns 10x slower to prevent "forgetting")
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-5}, 
        {'params': model.proj.parameters(), 'lr': 1e-4},
        {'params': classifier.parameters(), 'lr': 1e-4},
    ], weight_decay=0.05)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Helps with overlapping classes
    
    best_acc = 0.0
    patience = 12 # Slightly more patience for deep fine-tuning
    epochs_no_improve = 0
    
    #os.makedirs("models", exist_ok=True)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        val_acc = evaluate(model, classifier, val_loader, device)
        print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}")
        
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0

            prototypes = build_final_prototypes(model, train_loader, device)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'classifier_state': classifier.state_dict(),
                'prototypes': prototypes,
                'class_to_idx': train_ds.class_to_idx,
                'classes': train_ds.classes,
                'val_acc': val_acc
            }, "/home/abk/abk/projects/Major-project-basic-ui/models/new_convnext_mar11-2.pth")
            print(f"⭐ New Best Model: {val_acc:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("🛑 Early stopping triggered")
            break

    return best_acc

#### INCREMENTAL #######
def train_increment(data_root="/media/abk/New Disk/DATASETS/CLUSTER_INCREMENTAL_LEARNING", epochs=20, batch_size=8, device="cuda"):

    
    # Standard transforms (Keeping it simple for prototype)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load everything as training data
    train_ds = datasets.ImageFolder(data_root, train_transform)
    train_loader = DataLoader(
        train_ds, 
        batch_size=min(batch_size, len(train_ds)), 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    num_classes = len(train_ds.classes)
    print(f"🚀 Starting Prototype Training: {num_classes} classes, {len(train_ds)} total images.")

    # Initialize Model & Classifier
    model = ConvNeXtIncremental(embedding_dim=EMBEDDING_DIM).to(device)
    classifier = CosineClassifier(EMBEDDING_DIM, num_classes).to(device)
    
    # Fast learning for prototype
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-5}, 
        {'params': model.proj.parameters(), 'lr': 2e-4}, # Faster head learning
        {'params': classifier.parameters(), 'lr': 2e-4},
    ], weight_decay=0.01)
    
    criterion = nn.CrossEntropyLoss()
    save_path = "/home/abk/abk/projects/Major-project-basic-ui/models/convnext_incremental_weights.pth"

    model.train()
    classifier.train()

    for epoch in range(epochs):
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

    # Save the final state after all epochs
    torch.save({
        'model_state': model.state_dict(),
        'classifier_state': classifier.state_dict(),
        'classes': train_ds.classes,
        'class_to_idx': train_ds.class_to_idx
    }, save_path)
    
    print(f"✅ Prototype Model Saved to {save_path}")
    return True


if __name__ == "__main__":
    # Ensure this path matches your "New Disk" mount point
    DATA_PATH = "/media/abk/New Disk/DATASETS/first/updatedDataset"
    acc = train_model(DATA_PATH)
    print(acc)



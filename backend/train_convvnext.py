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
# 1. Metric-Learning Backbone (Improved)
# =====================================================
class ConvNeXtIncremental(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True, dropout=0.2):
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
            nn.Dropout(dropout),  # Added dropout for regularization
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        emb = self.proj(x)
        return F.normalize(emb, p=2, dim=1)


# =====================================================
# 2. Distance-Based Classifier (with learnable temperature)
# =====================================================
class CosineClassifier(nn.Module):
    def __init__(self, embedding_dim, n_classes):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_classes, embedding_dim))
        self.scale = nn.Parameter(torch.tensor(20.0))  # Learnable temperature
        
        # Initialize centers properly
        nn.init.kaiming_uniform_(self.centers, a=np.sqrt(5))

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
    for i, label in enumerate(unique_labels):
        prototypes[i] = embs[labs == label].mean(0)
    
    return F.normalize(prototypes, p=2, dim=1)


# =====================================================
# 4. Enhanced Training Loop
# =====================================================
def train_model(data_root, epochs=50, batch_size=32, lr=1e-4, device="cuda"):
    # Enhanced data augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # More aggressive cropping
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    # Initialize model and classifier
    model = ConvNeXtIncremental(embedding_dim=EMBEDDING_DIM, dropout=0.2).to(device)
    classifier = CosineClassifier(EMBEDDING_DIM, len(train_ds.classes)).to(device)
    
    # Different learning rates for different parts
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4},  # Higher for backbone
        {'params': model.proj.parameters(), 'lr': 2e-4},
        {'params': classifier.parameters(), 'lr': 2e-4},
    ], weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training tracking
    best_acc = 0.0
    patience = 10
    epochs_no_improve = 0
    min_delta = 0.001  # 0.1% minimum improvement
    
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        # Training phase
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation phase
        val_acc = evaluate(model, classifier, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}, LR: {current_lr:.2e}")
        
        # Learning rate scheduling
        scheduler.step()

        # Save best model (with classifier!)
        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            epochs_no_improve = 0
            
            # Build prototypes for the best model
            prototypes = build_final_prototypes(model, train_loader, device)
            
            # Save everything
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'classifier_state': classifier.state_dict(),
                'prototypes': prototypes,
                'class_to_idx': train_ds.class_to_idx,
                'classes': train_ds.classes,
                'val_acc': val_acc
            }, "models/convnext_best_weights.pth")
            print(f"⭐ New Best Model Saved (Acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping
        if epochs_no_improve >= patience:
            print("🛑 Early stopping triggered")
            break

    # Load best model and save final version
    print("\nTraining finished. Loading best model...")
    checkpoint = torch.load("models/convnext_best_weights.pth")
    model.load_state_dict(checkpoint['model_state'])
    
    # IMPORTANT: Load the classifier state too!
    classifier.load_state_dict(checkpoint['classifier_state'])
    
    # Build final prototypes
    final_prototypes = build_final_prototypes(model, train_loader, device)
    
    # Save final model with everything
    torch.save({
        'model_state': model.state_dict(),
        'classifier_state': classifier.state_dict(),
        'prototypes': final_prototypes,
        'class_to_idx': train_ds.class_to_idx,
        'classes': train_ds.classes,
        'val_acc': best_acc
    }, "models/convnext_final_for_ood.pth")
    
    print(f"✅ Final Model Saved with Accuracy: {best_acc:.4f}")
    return best_acc


def validate_final_model(data_root, model_path="models/convnext_final_for_ood.pth"):
    """Validate the saved model properly"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model and classifier
    model = ConvNeXtIncremental(embedding_dim=EMBEDDING_DIM).to(device)
    classifier = CosineClassifier(EMBEDDING_DIM, len(checkpoint['classes'])).to(device)
    
    # Load states
    model.load_state_dict(checkpoint['model_state'])
    classifier.load_state_dict(checkpoint['classifier_state'])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load validation data
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    # Evaluate
    acc = evaluate(model, classifier, val_loader, device)
    print(f"Final Model Validation Accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":
    DATA_PATH = "/media/abk/New Disk/DATASETS/first/updatedDataset"
    
    # Train the model
    best_acc = train_model(DATA_PATH, epochs=50)
    
    # Validate the final model
    validate_final_model(DATA_PATH)
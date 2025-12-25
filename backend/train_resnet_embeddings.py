# train_resnet_embeddings.py
import os
from pathlib import Path
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import torchvision.transforms as T

# -------------------------
# CBAM: Channel + Spatial Attention (lightweight)
# -------------------------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
    def forward(self, x):
        return self.bn(self.conv(x))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3,7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# -------------------------
# ResNet18-based embedding model
# -------------------------
class ResNet18Embedding(nn.Module):
    def __init__(self, embedding_dim=128, use_cbam=True, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        # remove final fc, keep up to avgpool
        modules = list(resnet.children())[:-2]  # up to last conv layer
        self.features = nn.Sequential(*modules)  # output shape: (B, 512, H/32, W/32)
        self.cbam = CBAM(512) if use_cbam else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )
        # init proj
        nn.init.xavier_uniform_(self.proj[0].weight)
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        emb = self.proj(x)
        emb = F.normalize(emb, p=2, dim=1)  # L2 normalize for cosine distances
        return emb

# -------------------------
# Simple dataset loader (image folder layout)
# data/
#   train/
#     classA/
#     classB/
#   val/
# -------------------------
def make_dataloaders(data_root, img_size=224, batch_size=32, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

def make_dataloaders_enhanced(
    data_root, 
    img_size=224, 
    batch_size=32, 
    num_workers=4,
    use_advanced_aug=True,
    augment_p=0.5  # Probability of applying augmentations
):
    """
    Enhanced dataloader with improved augmentations for better generalization.
    
    Args:
        data_root: Root directory with 'train' and 'val' subdirectories
        img_size: Target image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_advanced_aug: Whether to use advanced augmentations
        augment_p: Probability of applying each augmentation
    """
    
    # =========== TRAIN TRANSFORMS (ENHANCED) ===========
    if use_advanced_aug:
        train_transform = transforms.Compose([
            # Basic geometric augmentations
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),  # Small chance for vertical flip
            
            # Advanced geometric augmentations
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=10, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1), 
                shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            
            # Color augmentations
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            
            # Lighting and blur augmentations
            T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            
            # Noise and quality augmentations
            T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.1),
            T.RandomApply([T.RandomPosterize(bits=4)], p=0.1),
            T.RandomApply([T.RandomEqualize()], p=0.2),
            
            # Edge case augmentations
            T.RandomApply([T.RandomSolarize(threshold=192)], p=0.05),
            T.RandomApply([T.RandomAutocontrast()], p=0.2),
            
            # Normalization
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            # Optional: Add random erasing (cutout/mixup alternative)
            T.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        ])
    else:
        # Basic train transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # =========== VAL TRANSFORMS (ENHANCED) ===========
    # Multiple validation crops for better evaluation (like test-time augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),  # Slightly larger resize
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # =========== DATASETS ===========
    train_ds = datasets.ImageFolder(
        os.path.join(data_root, 'train'), 
        transform=train_transform
    )
    
    val_ds = datasets.ImageFolder(
        os.path.join(data_root, 'val'), 
        transform=val_transform
    )
    
    # =========== DATALOADERS ===========
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable batch norm
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, train_ds.classes
# -------------------------
# Classifier head for closed-set training (optional)
# -------------------------
class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, n_classes)
    def forward(self, emb):
        return self.fc(emb)

# -------------------------
# Prototype utilities (compute prototype per class)
# -------------------------
def compute_prototypes(embeddings, labels, num_classes):
    # embeddings: N x D torch tensor, labels: N
    prototypes = torch.zeros(num_classes, embeddings.size(1), device=embeddings.device)
    counts = torch.zeros(num_classes, device=embeddings.device)
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes[c] = embeddings[mask].mean(dim=0)
            counts[c] = mask.sum()
        else:
            prototypes[c] = torch.zeros(embeddings.size(1), device=embeddings.device)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes

# -------------------------
# OOD detection by cosine thresholding
# -------------------------
def detect_ood(emb, prototypes, threshold=0.6):
    # emb: B x D, prototypes: C x D; returns list of (is_known, best_class, best_score)
    cos = torch.matmul(emb, prototypes.t())  # B x C (since both normalized)
    best_scores, best_idx = cos.max(dim=1)
    is_known = best_scores >= threshold
    return is_known, best_idx, best_scores

# -------------------------
# Training loop (supervised base training)
# -------------------------
def train_supervised(embedding_net, classifier, train_loader, val_loader, classnames, epochs=10, lr=5e-4, patience=5, device='cuda'):

    embedding_net.to(device)
    classifier.to(device)
    params = list(embedding_net.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    ce = nn.CrossEntropyLoss()
    best_val = 0.0
    no_improve_epochs = 0
    min_delta = 1e-4  # minimum meaningful improvement
    for epoch in range(epochs):
        embedding_net.train(); classifier.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device); labels = labels.to(device)
            emb = embedding_net(imgs)
            logits = classifier(emb)
            loss = ce(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * imgs.size(0)
            _, preds = logits.max(1)
            correct += (preds==labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        train_acc = correct/total
        val_acc = evaluate_classifier(embedding_net, classifier, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/total:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val + min_delta:
            best_val = val_acc
            torch.save({
                'emb_state': embedding_net.state_dict(),
                'clf_state': classifier.state_dict(),
                'classes': classnames
            }, r'models/best_supervised.pth')
            no_improve_epochs = 0
        else:
            no_improve_epochs+=1
            print(f"No improvement for {no_improve_epochs} epochs")
        if no_improve_epochs>patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break   

    print("Best val acc:", best_val)

def evaluate_classifier(embedding_net, classifier, loader, device='cuda'):
    embedding_net.eval(); classifier.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device); labels = labels.to(device)
            emb = embedding_net(imgs)
            logits = classifier(emb)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return correct/total if total>0 else 0.0

# -------------------------
# Example of prototype-based recognition after embedding training
# -------------------------
def build_prototypes_from_loader(embedding_net, loader, n_classes, device='cuda'):
    embedding_net.eval()
    embs_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = embedding_net(imgs)
            embs_list.append(emb.cpu())
            labels_list.append(labels)
    embs = torch.cat(embs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    prototypes = compute_prototypes(embs, labels, n_classes)
    return prototypes

# -------------------------
# Exemplar memory for exemplar replay
# -------------------------
class ExemplarMemory:
    def __init__(self, max_images_per_class=20):
        self.storage = defaultdict(deque)  # class -> deque of (img, label)
        self.max = max_images_per_class
    def add(self, class_id, image_tensor, label):
        dq = self.storage[class_id]
        dq.append((image_tensor.cpu(), label))
        if len(dq) > self.max:
            dq.popleft()
    def sample_batch(self, batch_size):
        imgs = []
        labels = []
        keys = list(self.storage.keys())
        if not keys:
            return None, None
        for _ in range(batch_size):
            cls = np.random.choice(keys)
            img, lab = self.storage[cls][np.random.randint(len(self.storage[cls]))]
            imgs.append(img)
            labels.append(lab)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels, dtype=torch.long)
        return imgs, labels

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    data_root = "/media/abk/New Disk/DATASETS/first/updatedDataset"   # expects ./data/train and ./data/val
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparams
    embedding_dim = 128
    batch_size = 32
    lr = 5e-4
    epochs = 50
    min_delta = 1e-4  # minimum meaningful improvement

    # Make dataloaders
    train_loader, val_loader, classes = make_dataloaders(data_root, img_size=224, batch_size=batch_size)
    n_classes = len(classes)
    print("Classes:", classes)

    # Build models
    emb_net = ResNet18Embedding(embedding_dim=embedding_dim, use_cbam=True, pretrained=True)
    clf = ClassifierHead(embedding_dim, n_classes)

    # Train supervised base model
    train_supervised(emb_net, clf, train_loader, val_loader,classnames=classes, epochs=epochs, lr=lr, device=device)

    # Load best checkpoint
    ckpt = torch.load('models/best_supervised.pth', map_location=device)
    emb_net.load_state_dict(ckpt['emb_state'])
    clf.load_state_dict(ckpt['clf_state'])
    emb_net.to(device); emb_net.eval()

    # Build prototypes from train set (or exemplar set)
    prototypes = build_prototypes_from_loader(emb_net, train_loader, n_classes, device=device)
    # prototypes is on CPU (normalized). You can upload to device when detecting
    print("Built prototypes shape:", prototypes.shape)

    # OOD detection example:
    # Suppose we have a batch of crops `x_batch` (tensor)
    # x_batch = some_tensor.to(device)
    # emb = emb_net(x_batch)  # normalized
    # is_known, best_idx, best_score = detect_ood(emb.to(prototypes.device), prototypes.to(device), threshold=0.65)
    # process known/unknown accordingly

    # Example: create exemplar memory and add a few training images per class
    exemplar = ExemplarMemory(max_images_per_class=30)
    # To add, iterate over a few samples from the dataset
    cnt = 0
    for imgs, labels in train_loader:
        # add first 100 images
        for i in range(min(len(imgs), 5)):
            exemplar.add(int(labels[i].item()), imgs[i], int(labels[i].item()))
        cnt += imgs.size(0)
        if cnt > 200:
            break

    # Save final models
    torch.save({
        'emb_state': emb_net.state_dict(),
        'clf_state': clf.state_dict(),
        'classes': classes
    }, r'models/final_model.pth')
    print("Saved final_model.pth")

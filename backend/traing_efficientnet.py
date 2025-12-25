import os
from pathlib import Path
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as T

# ======================================================
# EfficientNet-V2-S Embedding Backbone (REPLACEMENT)
# ======================================================
class EfficientNetV2Embedding(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()

        weights = (
            torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        backbone = torchvision.models.efficientnet_v2_s(weights=weights)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = backbone.classifier[1].in_features  # 1280

        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        emb = self.proj(x)
        return F.normalize(emb, dim=1)


# ======================================================
# Classifier Head (UNCHANGED)
# ======================================================
class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, n_classes)

    def forward(self, emb):
        return self.fc(emb)


# ======================================================
# DATALOADERS (UNCHANGED – you can keep enhanced version)
# ======================================================
def make_dataloaders(data_root, img_size=224, batch_size=32, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_root, 'train'),
                                    transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_root, 'val'),
                                  transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader, train_ds.classes


# ======================================================
# PROTOTYPES (UNCHANGED)
# ======================================================
def compute_prototypes(embeddings, labels, num_classes):
    prototypes = torch.zeros(num_classes, embeddings.size(1))
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            prototypes[c] = embeddings[mask].mean(0)
    return F.normalize(prototypes, dim=1)


def build_prototypes_from_loader(embedding_net, loader, n_classes, device):
    embedding_net.eval()
    embs, labs = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = embedding_net(x)
            embs.append(emb.cpu())
            labs.append(y)

    embs = torch.cat(embs)
    labs = torch.cat(labs)

    return compute_prototypes(embs, labs, n_classes)


# ======================================================
# TRAINING LOOP (UNCHANGED)
# ======================================================
def train_supervised(embedding_net, classifier,
                     train_loader, val_loader,
                     classnames, epochs=40,
                     lr=3e-4, patience=7,
                     device='cuda'):

    embedding_net.to(device)
    classifier.to(device)

    opt = torch.optim.AdamW(
        list(embedding_net.parameters()) +
        list(classifier.parameters()),
        lr=lr, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = 0
    no_improve = 0

    for epoch in range(epochs):
        embedding_net.train()
        classifier.train()

        total, correct, loss_sum = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            emb = embedding_net(x)
            logits = classifier(emb)

            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_acc = correct / total

        val_acc = evaluate_classifier(
            embedding_net, classifier, val_loader, device
        )

        print(f"Epoch {epoch+1}/{epochs} "
              f"loss={loss_sum/total:.4f} "
              f"train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            torch.save({
                'emb_state': embedding_net.state_dict(),
                'clf_state': classifier.state_dict(),
                'classes': classnames
            }, "models/efficentnet_best_supervised.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break


def evaluate_classifier(embedding_net, classifier, loader, device):
    embedding_net.eval()
    classifier.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            emb = embedding_net(x)
            preds = classifier(emb).argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return correct / total


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    data_root = "/media/abk/New Disk/DATASETS/first/updatedDataset"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_dim = 256
    batch_size = 32
    epochs = 40

    train_loader, val_loader, classes = make_dataloaders(
        data_root, batch_size=batch_size
    )

    emb_net = EfficientNetV2Embedding(
        embedding_dim=embedding_dim, pretrained=True
    )
    clf = ClassifierHead(embedding_dim, len(classes))

    train_supervised(
        emb_net, clf,
        train_loader, val_loader,
        classes, device=device
    )

    ckpt = torch.load("models/best_supervised.pth", map_location=device)
    emb_net.load_state_dict(ckpt["emb_state"])
    clf.load_state_dict(ckpt["clf_state"])

    prototypes = build_prototypes_from_loader(
        emb_net, train_loader, len(classes), device
    )

    torch.save({
        'emb_state': emb_net.state_dict(),
        'clf_state': clf.state_dict(),
        'prototypes': prototypes,
        'classes': classes
    }, "models/efficentnet_final_model.pth")

    print("✅ EfficientNet-V2 model trained & saved")

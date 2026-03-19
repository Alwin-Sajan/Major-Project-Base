import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from PIL import Image

# Use 768 for future-proofing and ConvNeXt-Tiny native compatibility
EMBEDDING_DIM = 768 

# =====================================================
# 1. Hierarchical Taxonomic Dataset
# =====================================================
class TaxonomicDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        self.families, self.genera, self.species_classes = set(), set(), set()
        
        # Walk through: Family/Genus/Species
        for family in sorted(os.listdir(root_dir)):
            fam_path = os.path.join(root_dir, family)
            if not os.path.isdir(fam_path): continue
            for genus in sorted(os.listdir(fam_path)):
                gen_path = os.path.join(fam_path, genus)
                if not os.path.isdir(gen_path): continue
                for species in sorted(os.listdir(gen_path)):
                    spec_path = os.path.join(gen_path, species)
                    if not os.path.isdir(spec_path): continue
                    for img_name in os.listdir(spec_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append({
                                "path": os.path.join(spec_path, img_name),
                                "family": family, "genus": genus, "species": species
                            })
                            self.families.add(family)
                            self.genera.add(genus)
                            self.species_classes.add(species)

        self.families = sorted(list(self.families))
        self.genera = sorted(list(self.genera))
        self.species_classes = sorted(list(self.species_classes))
        
        self.fam_to_idx = {name: i for i, name in enumerate(self.families)}
        self.gen_to_idx = {name: i for i, name in enumerate(self.genera)}
        self.spec_to_idx = {name: i for i, name in enumerate(self.species_classes)}
        self.targets = [self.spec_to_idx[s["species"]] for s in self.samples]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['path']).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, {
            "family": torch.tensor(self.fam_to_idx[item['family']], dtype=torch.long),
            "genus": torch.tensor(self.gen_to_idx[item['genus']], dtype=torch.long),
            "species": torch.tensor(self.spec_to_idx[item['species']], dtype=torch.long)
        }

# =====================================================
# 2. Model & Multi-Head Classifier
# =====================================================
class TaxonomyConvNeXt(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = torchvision.models.convnext_tiny(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(768, 768), 
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, embedding_dim),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.proj(x), p=2, dim=1)

class MultiHeadCosineClassifier(nn.Module):
    def __init__(self, embedding_dim, n_fam, n_gen, n_spec):
        super().__init__()
        self.fam_centers = nn.Parameter(torch.randn(n_fam, embedding_dim))
        self.gen_centers = nn.Parameter(torch.randn(n_gen, embedding_dim))
        self.spec_centers = nn.Parameter(torch.randn(n_spec, embedding_dim))
        self.scale = nn.Parameter(torch.tensor(20.0))
        nn.init.kaiming_uniform_(self.fam_centers, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.gen_centers, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.spec_centers, a=np.sqrt(5))

    def forward(self, emb):
        return {
            "family": torch.mm(emb, F.normalize(self.fam_centers, p=2, dim=1).t()) * self.scale,
            "genus": torch.mm(emb, F.normalize(self.gen_centers, p=2, dim=1).t()) * self.scale,
            "species": torch.mm(emb, F.normalize(self.spec_centers, p=2, dim=1).t()) * self.scale
        }

# =====================================================
# 3. Training & Incremental Logic
# =====================================================
def get_sampler(ds):
    counts = np.bincount(ds.targets)
    weights = 1.0 / (counts + 1e-6)
    sample_weights = torch.DoubleTensor(weights[ds.targets])
    return WeightedRandomSampler(sample_weights, len(sample_weights))

@torch.no_grad()
def build_prototypes(model, ds, device):
    model.eval()
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    all_embs, all_labs = [], []
    for x, y in loader:
        all_embs.append(model(x.to(device)).cpu())
        all_labs.append(y["species"])
    embs, labs = torch.cat(all_embs), torch.cat(all_labs)
    protos = torch.stack([embs[labs == i].mean(0) for i in range(len(ds.species_classes))])
    return F.normalize(protos, p=2, dim=1)

def train_incremental(data_root, checkpoint_path=None, epochs=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Datasets
    t_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),  # Useful if fish are captured at odd angles/swimming down
        transforms.RandomRotation(15),         # Slightly reduced to prevent "extreme" tilts
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),     # Force model to learn shape/patterns, not just color
        transforms.RandomAffine(
            degrees=0,                         # Rotation is already handled above
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1)                   # Tighter scaling to preserve detail
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    v_trans = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = TaxonomicDataset(os.path.join(data_root, "train"), t_trans)
    val_ds = TaxonomicDataset(os.path.join(data_root, "val"), v_trans)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=get_sampler(train_ds), num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # Initialize Model
    model = TaxonomyConvNeXt().to(device)
    classifier = MultiHeadCosineClassifier(EMBEDDING_DIM, len(train_ds.families), len(train_ds.genera), len(train_ds.species_classes)).to(device)
    
    # INCREMENTAL SEEDING LOGIC
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"--- Loading and Seeding Old Model ---")
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state'])
        
        # Map old species centers to new index positions
        old_spec_to_idx = ckpt['class_to_idx']
        old_centers = ckpt['classifier_state']['spec_centers']
        with torch.no_grad():
            for name, old_idx in old_spec_to_idx.items():
                if name in train_ds.spec_to_idx:
                    new_idx = train_ds.spec_to_idx[name]
                    classifier.spec_centers[new_idx].copy_(old_centers[old_idx])

    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-6 if checkpoint_path else 1e-5},
        {'params': model.proj.parameters(), 'lr': 1e-4},
        {'params': classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=0.05)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train(); classifier.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), {k: v.to(device) for k, v in y.items()}
            logits = classifier(model(x))
            loss = criterion(logits["family"], y["family"]) + criterion(logits["genus"], y["genus"]) + (2.0 * criterion(logits["species"], y["species"]))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Eval & Save
        model.eval(); classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = classifier(model(x.to(device)))
                correct += (logits["species"].argmax(1) == y["species"].to(device)).sum().item()
                total += x.size(0)
        
        val_acc = correct / total
        print(f"Validation Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            protos = build_prototypes(model, TaxonomicDataset(os.path.join(data_root, "train"), v_trans), device)
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(), 'classifier_state': classifier.state_dict(),
                'prototypes': protos, 'class_to_idx': train_ds.spec_to_idx, 'classes': train_ds.species_classes, 'val_acc': val_acc
            }, f"models/taxobot_latest.pth")

if __name__ == "__main__":
    # If it's your FIRST time: checkpoint_path=None
    # If you are ADDING species: checkpoint_path="models/taxobot_latest.pth"
    train_incremental(data_root="/media/abk/New Disk/DATASETS/marine_v2", checkpoint_path="models/taxobot_latest.pth")
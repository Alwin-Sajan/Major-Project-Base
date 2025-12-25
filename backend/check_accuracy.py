import os
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. Model Definition (Must match training structure)
# =====================================================
class ConvNeXtIncremental(torch.nn.Module):
    def __init__(self, embedding_dim=256, pretrained=False):
        super().__init__()
        # Weights=None because we are loading custom weights
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'convnext_tiny', weights=None)
        
        self.features = backbone.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        
        in_features = backbone.classifier[2].in_features 
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        emb = self.proj(x)
        return F.normalize(emb, p=2, dim=1)

# =====================================================
# 2. Evaluation Logic
# =====================================================
def run_evaluation(data_root, model_path, device="cuda"):
    # --- Setup ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load Model
    model = ConvNeXtIncremental(embedding_dim=256).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Load Prototypes and Classes
    prototypes = checkpoint['prototypes'].to(device)
    class_names = checkpoint['classes']
    
    # Check consistency
    assert len(class_names) == len(val_ds.classes), \
        "Mismatch between training classes and validation folder classes!"
    
    # --- Evaluation Loop ---
    correct = 0
    total = 0
    
    # Store similarities for analysis
    correct_sims = []
    incorrect_sims = []
    
    print("Running inference...")
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x, y = x.to(device), y.to(device)
            
            # 1. Get Embedding
            emb = model(x)
            
            # 2. Compare to Prototypes (Cosine Similarity)
            # Shapes: emb [B, 256] x prototypes [N_classes, 256].T -> [B, N_classes]
            similarities = torch.mm(emb, prototypes.t())
            
            # 3. Get Prediction (Class with max similarity)
            max_sim, preds = similarities.max(dim=1)
            
            # 4. Stats
            match_mask = (preds == y)
            correct += match_mask.sum().item()
            total += x.size(0)
            
            # Store confidence scores for analytics
            correct_sims.extend(max_sim[match_mask].cpu().numpy())
            incorrect_sims.extend(max_sim[~match_mask].cpu().numpy())

    # --- Results ---
    acc = correct / total
    print("\n" + "="*40)
    print(f"FINAL ACCURACY: {acc*100:.2f}%")
    print("="*40)
    
    # --- OOD Calibration Analysis ---
    if len(correct_sims) > 0:
        avg_conf_correct = np.mean(correct_sims)
        print(f"Avg Confidence (Correct Preds):   {avg_conf_correct:.4f}")
        
    if len(incorrect_sims) > 0:
        avg_conf_incorrect = np.mean(incorrect_sims)
        print(f"Avg Confidence (Incorrect Preds): {avg_conf_incorrect:.4f}")
        print(f"-> Suggested OOD Threshold:       {(avg_conf_correct + avg_conf_incorrect)/2:.4f}")
    else:
        print("Perfect accuracy! No incorrect predictions to analyze for OOD threshold.")
        print(f"-> Suggested OOD Threshold:       {np.mean(correct_sims) - 0.1:.4f}")

    # Plot histogram for visualization
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(correct_sims, bins=50, alpha=0.7, label='Correct Predictions', color='green')
        if len(incorrect_sims) > 0:
            plt.hist(incorrect_sims, bins=50, alpha=0.7, label='Incorrect Predictions', color='red')
        plt.xlabel("Cosine Similarity Score")
        plt.ylabel("Count")
        plt.title("Confidence Distribution (Calibration for OOD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("ood_calibration_plot.png")
        print("Saved calibration plot to 'ood_calibration_plot.png'")
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == "__main__":
    # Update paths as needed
    DATA_PATH = "/media/abk/New Disk/DATASETS/first/updatedDataset"
    MODEL_PATH = "models/convnext_final_il_ready.pth"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_evaluation(DATA_PATH, MODEL_PATH, device)
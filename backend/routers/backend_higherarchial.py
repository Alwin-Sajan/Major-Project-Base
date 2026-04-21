import io
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import APIRouter, UploadFile, File
import os

# Import your classes and utility functions
# Make sure TaxonomyConvNeXt and MultiHeadCosineClassifier are accessible
from train_convnext_higherarchial import TaxonomyConvNeXt, MultiHeadCosineClassifier, EMBEDDING_DIM
import torchvision

backend_router_high = APIRouter()

# =====================================================
# 1. Global Initialization
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/home/abk/abk/projects/Major-project-basic-ui/models/taxobot_hierarchical_mar21.pth"
DATA_ROOT = "/media/abk/New Disk/DATASETS/marine_v2/train"

# We need to recreate the class lists in the EXACT SAME ORDER as training
def get_taxonomy_labels(root_dir):
    families, genera, species = set(), set(), set()
    for f in sorted(os.listdir(root_dir)):
        fam_p = os.path.join(root_dir, f)
        if not os.path.isdir(fam_p): continue
        families.add(f)
        for g in sorted(os.listdir(fam_p)):
            gen_p = os.path.join(fam_p, g)
            if not os.path.isdir(gen_p): continue
            genera.add(g)
            for s in sorted(os.listdir(gen_p)):
                if os.path.isdir(os.path.join(gen_p, s)):
                    species.add(s)
    return sorted(list(families)), sorted(list(genera)), sorted(list(species))

family_names, genus_names, species_names = get_taxonomy_labels(DATA_ROOT)

# Load Model
ckpt = torch.load(MODEL_PATH, map_location=device)
embedding_net = TaxonomyConvNeXt(EMBEDDING_DIM).to(device)
classifier = MultiHeadCosineClassifier(
    EMBEDDING_DIM, len(family_names), len(genus_names), len(species_names)
).to(device)

embedding_net.load_state_dict(ckpt['model_state'])
classifier.load_state_dict(ckpt['classifier_state'])
prototypes = ckpt['prototypes'].to(device) # [Num_Species, 768]

embedding_net.eval()
classifier.eval()

# Inference Transform (Must match v_trans)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================================================
# 2. The Endpoint
# =====================================================
@backend_router_high.post("/predictHierarchy")
async def predictHierarchy(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # 1. Get Embedding and Multi-Head Logits
        emb = embedding_net(img_tensor) # forward returns normalized emb
        logits = classifier(emb)
        
        # 2. Taxonomic Head Predictions (Highest logit)
        fam_idx = logits["family"].argmax(1).item()
        gen_idx = logits["genus"].argmax(1).item()
        
        # 3. Species OOD Logic using Prototypes (Best Score & Margin)
        # mm calculates cosine similarity because emb and prototypes are normalized
        cosine_scores = torch.mm(emb, prototypes.t()) 
        best_score, best_idx = cosine_scores.max(dim=1)
        
        top2 = torch.topk(cosine_scores, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).item()
        score = best_score.item()

        # 4. OOD Logic
        if score < 0.65   or margin < 0.15 :
            # Re-calculate for storage if needed, or use current emb
            # utils.store_unknown(
            #     image=image,
            #     embedding=emb.cpu().numpy(),
            #     confidence=score
            # )
            
            return {
                "class_name": "UNKNOWN",
                "confidence": 0.01,
                "ood": True,
                "hierarchy": "Unknown"
            }

        # 5. Success Result
        return {
            "family": family_names[fam_idx],
            "genus": genus_names[gen_idx],
            "species": species_names[best_idx.item()],
            "confidence": round(score * 100, 2),
            "margin": round(margin, 4),
            "ood": False
        }
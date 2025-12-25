import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision

# =========================================================
# IMPORT MODEL (MATCH TRAINING CODE)
# =========================================================
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

# =========================================================
# FastAPI setup
# =========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load model + prototypes
# =========================================================
MODEL_PATH = "../models/convnext_final_for_ood.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["classes"]
num_classes = len(class_names)

# ---- Embedding model ----
embedding_net = ConvNeXtIncremental(
    embedding_dim=256,
    pretrained=False
)

embedding_net.load_state_dict(checkpoint["model_state"])
embedding_net.to(device).eval()

# ---- Prototypes ----
prototypes = checkpoint["prototypes"].to(device)  # already normalized

# =========================================================
# Image preprocessing (MUST match training)
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# OOD hyperparameters (TUNE THESE)
# =========================================================
OOD_THRESHOLD = 0.65     # cosine similarity
MARGIN_THRESHOLD = 0.15 # top1 - top2 gap

# =========================================================
# Prediction endpoint (OPEN-SET)
# =========================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # 1. Extract embedding
        emb = embedding_net(img_tensor)
        emb = F.normalize(emb, p=2, dim=1)

        # 2. Cosine similarity to prototypes
        cosine_scores = torch.matmul(emb, prototypes.T)
        best_score, best_idx = cosine_scores.max(dim=1)

        # Debug logs (keep for now)
        print("Cosine scores:", cosine_scores.cpu().numpy())
        print("Best score:", best_score.item())

        # 3. Margin-based confidence
        top2 = torch.topk(cosine_scores, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).item()

        score = best_score.item()

        # 4. OOD decision
        if score < OOD_THRESHOLD or margin < MARGIN_THRESHOLD:
            return {
                "class_name": "UNKNOWN",
                "confidence": round(score * 100, 2),
                "ood": True
            }
        else:
            return {
                "class_name": class_names[best_idx.item()],
                "confidence": round(score * 100, 2),
                "ood": False
            }

# =========================================================
# Run server
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import io
import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import utils


# =========================================================
# FastAPI setup
# =========================================================

backend_router = APIRouter()

# =========================================================
# Load model + prototypes
# =========================================================
device = utils.device
embedding_net = utils.embedding_net
prototypes = utils.prototypes
class_names = utils.class_names


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
# Prediction endpoint 
# =========================================================
@backend_router.post("/predictSimpleDetection")
async def predictSimple(file: UploadFile = File(...)):
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

        # 3. Margin-based confidence
        top2 = torch.topk(cosine_scores, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).item()
        score = best_score.item()

        print("Margin:", margin)
        print("Best score:", best_score.item())

        # 4. OOD decision
        print(margin)
        if score < utils.OOD_THRESHOLD or margin < utils.MARGIN_THRESHOLD:
            return {
                "class_name": "UNKNOWN",
                "confidence": 0.01,
                "ood": True
            }
        else:
            return {
                "class_name": class_names[best_idx.item()],
                "confidence": round(score * 100, 2),
                "ood": False
            }
# =========================================================

@backend_router.post("/predictOODDetection")
async def predictOOD(file: UploadFile = File(...)):
    print("ood backnd")
    #run_clustering()

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = embedding_net(img_tensor)
        emb = F.normalize(emb, p=2, dim=1)

        cosine_scores = torch.matmul(emb, prototypes.T)
        best_score, best_idx = cosine_scores.max(dim=1)

        top2 = torch.topk(cosine_scores, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).item()
        score = best_score.item()

        if score < utils.OOD_THRESHOLD or margin < utils.MARGIN_THRESHOLD:
            # ---- STORE UNKNOWN ----
            default_embed = utils.embedding_PRETRAINED_DEFAULT
            emb = default_embed(img_tensor)
            emb = F.normalize(emb, p=2, dim=1)
            
            utils.store_unknown(
                image=image,
                embedding=emb.cpu().numpy(),
                confidence=score
            )

            # ---- CHECK TRIGGERS ----
            print("clustering")
            do_cluster, reason = utils.checkClusterCondition()
            if do_cluster:
                utils.run_clustering()

            return {
                "class_name": "UNKNOWN",
                "confidence": 0.01,
                "ood": True
            }

        return {
            "class_name": class_names[best_idx.item()],
            "confidence": round(score * 100, 2),
            "ood": False
        }


# =========================================================



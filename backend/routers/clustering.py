import os
import torch
import json
import utils
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from fastapi import HTTPException, Body
from fastapi import APIRouter


# --- CONFIGURATION ---
device = utils.device
embedding_net = utils.embedding_net
prototypes = utils.prototypes
class_names = utils.class_names

clustering_router = APIRouter()
#------------------------
IMG_DIR =  r"/media/abk/New Disk/DATASETS/clusterdataset" #NOTE:utils.IMG_DIR, CHANGE IT IN FUTURE< ONLY FOR TESTING

# ================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_data():
    if not os.path.exists(utils.CLUSTER_META_PATH):
        return {}, []
    with open(utils.CLUSTER_META_PATH, "r") as f:
        clusters = json.load(f)
    all_images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return clusters, all_images




# ==============================
@clustering_router.get("/sampleclusterTest")
async def runcluster():
    print("clustering api works")
    return {None}



@clustering_router.get("/runcluster")
async def runcluster():
    print("staring clustering")
    #utils.run_clustering()
    utils.run_clustering()
    print("ending clustering")
    return {None}


@clustering_router.get("/testing")
async def testing():
    import os
    count = 0
    PATH = r"/media/abk/New Disk/DATASETS/clusterdataset" 
    for image_name in os.listdir(PATH):
        
        full_path = os.path.join(PATH,image_name)
        image = Image.open(full_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            default_embed = utils.embedding_PRETRAINED_DEFAULT
            emb = default_embed(img_tensor)
            emb = F.normalize(emb, p=2, dim=1) #
            cosine_scores = torch.matmul(emb, prototypes.T)
            best_score, best_idx = cosine_scores.max(dim=1)
            utils.store_unknown(
                image=image,
                embedding=emb.cpu().numpy(),
                confidence=best_score.item()    
            )
            count+=1

    #Cluster update
    print("staring clustering")
    utils.run_clustering()
    print("ending clustering")

    return {count}


@clustering_router.get("/testingifAnImageisKNOWN")
async def testingifAnImageisKNOWN():
    import os

    PATH = r"/media/abk/New Disk/DATASETS/clusterdataset" 
    for image_name in os.listdir(PATH):
        full_path = os.path.join(PATH,image_name)
        image = Image.open(full_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = utils.embedding_net(img_tensor)
            emb = F.normalize(emb, p=2, dim=1)
            cosine_scores = torch.matmul(emb, prototypes.T)
            best_score, best_idx = cosine_scores.max(dim=1)
            top2 = torch.topk(cosine_scores, k=2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).item()
            score = best_score.item()
            if score < utils.OOD_THRESHOLD or margin < utils.MARGIN_THRESHOLD:
                ...
            else:
                print(f"{class_names[best_idx.item()]} ({round(score * 100, 2)}) -- ",full_path)

    return {None}



@clustering_router.get("/clusters")
def get_all_clusters():
    """Returns a list of clusters with a 'cover' image for the UI"""
    clusters, all_images = get_data() 
    summary = []

    clusters = clusters["clusters"]
    
    for c_id, indices in clusters.items():
        if not indices: continue
        cover_img = all_images[indices[0]]
        summary.append({
            "id": c_id,
            "count": len(indices),
            "preview_url": f"/images/{cover_img}"
        })
    return summary

@clustering_router.get("/clusters/{cluster_id}")
def get_cluster_details(cluster_id: str):
    """Returns all image URLs for a specific cluster"""
    clusters, all_images = get_data()

    clusters = clusters["clusters"]
    
    if cluster_id not in clusters:
        raise HTTPException(status_code=404, detail="Cluster not found")
        
    indices = clusters[cluster_id]

    print("CLUSTER:", cluster_id)
    print("INDICES:", indices)
    print("ALL_IMAGES_LEN:", len(all_images))

    valid_indices = [i for i in indices if 0 <= i < len(all_images)]

    image_urls = [
        { "id": i, "url": f"/images/{all_images[i]}" }
        for i in valid_indices
    ]

    

    
    return {
        "cluster": cluster_id,
        "total": len(indices),
        "images": image_urls
    }

@clustering_router.patch("/clusters/{cluster_id}")
def update_cluster_name(cluster_id: str, new_name: str = Body(..., embed=True)):
    with open(utils.CLUSTER_META_PATH, "r") as f:
        clusters = json.load(f)
    
    #clusters = clusters["clusters"]

    if cluster_id not in clusters["clusters"]:
        raise HTTPException(status_code=404, detail="Cluster not found")

    # Update clusters mapping
    clusters["clusters"][new_name] = clusters["clusters"].pop(cluster_id)
    
    # Update cluster_info mapping if it exists
    if "cluster_info" in clusters["clusters"] and cluster_id in clusters["cluster_info"]:
        clusters["clusters"]["cluster_info"][new_name] = clusters["clusters"]["cluster_info"].pop(cluster_id)

    with open(utils.CLUSTER_META_PATH, "w") as f:
        json.dump(clusters, f, indent=2)

    return {"message": "Updated successfully", "new_id": new_name}


@clustering_router.post("/clusters/move")
def move_images(source_id: str = Body(...),target_id: str = Body(...),indices: list[int] = Body(...)):
    with open(utils.CLUSTER_META_PATH, "r") as f:
        data = json.load(f)

    # Convert indices to match JSON types (usually ints)
    indices_to_move = [int(i) for i in indices]

    # 1. Update the CLUSTERS list (The actual image movement)
    # Ensure keys exist to avoid crashes
    if source_id not in data["clusters"]:
        return {"error": f"Source cluster {source_id} not found"}, 404
    
    if target_id not in data["clusters"]:
         # Create target if it doesn't exist (optional, but prevents crash)
        data["clusters"][target_id] = []

    # Remove from source
    data["clusters"][source_id] = [
        i for i in data["clusters"][source_id] if i not in indices_to_move
    ]
    
    existing = set(data["clusters"][target_id])
    to_add = [i for i in indices_to_move if i not in existing]
    data["clusters"][target_id].extend(to_add)

    # 2. Safely Update CLUSTER_INFO (The metadata)
    # Check if cluster_info exists AND if the specific cluster keys exist inside it
    if "cluster_info" in data:
        if source_id in data["cluster_info"]:
            data["cluster_info"][source_id]["size"] = len(data["clusters"][source_id])
        
        if target_id in data["cluster_info"]:
            data["cluster_info"][target_id]["size"] = len(data["clusters"][target_id])
        else:
            # Optional: If target has no info, create a basic entry
            data["cluster_info"][target_id] = {
                "size": len(data["clusters"][target_id]),
                "name": target_id
            }

    with open(utils.CLUSTER_META_PATH, "w") as f:
        json.dump(data, f, indent=2)

    return {"message": "Moved successfully"}
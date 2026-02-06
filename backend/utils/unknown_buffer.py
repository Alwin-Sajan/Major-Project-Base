import os, json, time
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from . import config

os.makedirs(config.IMG_DIR, exist_ok=True)

# CLUSTERING STATS
def cluster_stats(labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"CLUSTERING RESULT")
    print(f"------------------------------------------------")
    print(f"Total Images: {len(labels)}")
    print(f"Clusters Found: {n_clusters}")
    print(f"Noise Images: {n_noise}")
    print(f"Labels distribution: {unique_labels}")
    print(f"------------------------------------------------")

# ------------------------------------------------
# Store unknown sample
# ------------------------------------------------
def store_unknown(image: Image.Image, embedding: np.ndarray, confidence: float):
    from uuid import uuid4
    ts = time.strftime("%Y_%m_%d_%H_%M_%S")
    fname = f"{ts}_{uuid4().hex}.jpg"

    # 1. Save image NOTE
    #image.save(os.path.join(IMG_DIR, fname))

    # 2. Save embedding
    if os.path.exists(config.EMB_PATH):
        old = np.load(config.EMB_PATH)
        new = np.vstack([old, embedding])
    else:
        new = embedding
    np.save(config.EMB_PATH, new)

    # 3. Save metadata
    meta = []
    if os.path.exists(config.META_PATH):
        meta = json.load(open(config.META_PATH))

    meta.append({
        "file": fname,
        "confidence": float(confidence),
        "timestamp": ts
    })

    json.dump(meta, open(config.META_PATH, "w"), indent=2)

# ------------------------------------------------
# Trigger checks
# ------------------------------------------------
def checkClusterCondition():
    # Trigger 1: count
    if os.path.exists(config.EMB_PATH):
        count = np.load(config.EMB_PATH).shape[0]
        if count >= config.UNKNOWN_COUNT_THRESHOLD:
            return True, "COUNT_TRIGGER"

    # Trigger 2: time
    now = time.time()
    if os.path.exists(config.LAST_CLUSTER_TIME_PATH):
        last = float(open(config.LAST_CLUSTER_TIME_PATH).read())
        if now - last >= config.CLUSTER_TIME_THRESHOLD:
            return True, "TIME_TRIGGER"

    return False, None


# CLUSTERING
def run_clustering():
    if not os.path.exists(config.EMB_PATH):
        return

    embs = np.load(config.EMB_PATH)
    embs = normalize(embs, axis=1) 

    # DBSCAN works well for unknown discovery
    clusterer = DBSCAN(
        eps=0.17,min_samples=5,
        metric="cosine"
    )

    labels = clusterer.fit_predict(embs)
    cluster_stats(labels)

    # Save cluster assignments
    clusters = {}
    for idx, lbl in enumerate(labels):
        if lbl == -1:
            continue  # noise
        clusters.setdefault(f"cluster_{lbl}", []).append(idx)

    json.dump(clusters, open(config.CLUSTER_META_PATH, "w"), indent=2)

    update_last_cluster_time()

def update_last_cluster_time():
    open(config.LAST_CLUSTER_TIME_PATH, "w").write(str(time.time()))


def run_clustering_fixed():
    if not os.path.exists(config.EMB_PATH):
        print("No embeddings found.")
        return
    
    # 1. Load Embeddings
    embs = np.load(config.EMB_PATH)
    print(f"Original shape: {embs.shape}")

    # 2. Normalize FIRST (Critical for Cosine-like behavior)
    embs = normalize(embs, norm='l2')

    # 3. APPLY PCA (The Magic Fix)
    # Reduces 1024 dims -> 50 dims. 
    # This removes "background noise" (like blue water) from dominating the distance.
    pca = PCA(n_components=50) # 50 is usually the sweet spot for species
    embs_pca = pca.fit_transform(embs)
    
    # 4. Normalize AGAIN after PCA
    # This ensures Euclidean distance in PCA space = Cosine distance
    embs_pca = normalize(embs_pca, norm='l2')

    # 5. Run DBSCAN with TIGHTER settings
    # Note: eps values are different for PCA-reduced data.
    # Start at 0.4 and tweak. Since we normalized, max distance is 2.0.
    clusterer = DBSCAN(
        eps=0.5,       # Try 0.35 if still mixing, 0.50 if too much noise
        min_samples=5,  # Keep this low to catch small species groups
        metric="euclidean"
    )
    
    labels = clusterer.fit_predict(embs_pca)
    cluster_stats(labels)
    
    # 6. Save (Same as before)
    clusters = {}
    for idx, lbl in enumerate(labels):
        if lbl == -1:
            continue
        clusters.setdefault(f"cluster_{lbl}", []).append(idx)
        
    json.dump(clusters, open(config.CLUSTER_META_PATH, "w"), indent=2)
    print("Saved clusters.json")
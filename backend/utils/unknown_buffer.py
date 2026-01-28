import os, json, time
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from . import config

os.makedirs(config.IMG_DIR, exist_ok=True)

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
        eps=0.2,min_samples=3,
        metric="cosine"
    )

    labels = clusterer.fit_predict(embs) ;print(labels)

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


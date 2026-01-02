import os, json, time
import numpy as np
from PIL import Image
from .config import *

os.makedirs(IMG_DIR, exist_ok=True)

# ------------------------------------------------
# Store unknown sample
# ------------------------------------------------
def store_unknown(image: Image.Image, embedding: np.ndarray, confidence: float):
    ts = time.strftime("%Y_%m_%d_%H_%M_%S")
    fname = f"{ts}.jpg"

    # 1. Save image
    image.save(os.path.join(IMG_DIR, fname))

    # 2. Save embedding
    if os.path.exists(EMB_PATH):
        old = np.load(EMB_PATH)
        new = np.vstack([old, embedding])
    else:
        new = embedding

    np.save(EMB_PATH, new)

    # 3. Save metadata
    meta = []
    if os.path.exists(META_PATH):
        meta = json.load(open(META_PATH))

    meta.append({
        "file": fname,
        "confidence": float(confidence),
        "timestamp": ts
    })

    json.dump(meta, open(META_PATH, "w"), indent=2)

# ------------------------------------------------
# Trigger checks
# ------------------------------------------------
def checkClusterCondition():
    # Trigger 1: count
    if os.path.exists(EMB_PATH):
        count = np.load(EMB_PATH).shape[0]
        if count >= UNKNOWN_COUNT_THRESHOLD:
            return True, "COUNT_TRIGGER"

    # Trigger 2: time
    now = time.time()
    if os.path.exists(LAST_CLUSTER_TIME_PATH):
        last = float(open(LAST_CLUSTER_TIME_PATH).read())
        if now - last >= CLUSTER_TIME_THRESHOLD:
            return True, "TIME_TRIGGER"

    return False, None

def update_last_cluster_time():
    open(LAST_CLUSTER_TIME_PATH, "w").write(str(time.time()))

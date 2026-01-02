import json, os
import numpy as np
from sklearn.cluster import DBSCAN
from utils.config import *
from utils.unknown_buffer import update_last_cluster_time

def run_clustering():
    if not os.path.exists(EMB_PATH):
        return

    embs = np.load(EMB_PATH)

    # DBSCAN works well for unknown discovery
    clusterer = DBSCAN(
        eps=0.45,
        min_samples=2,
        metric="cosine"
    )

    labels = clusterer.fit_predict(embs) ;print(labels)

    # Save cluster assignments
    clusters = {}
    for idx, lbl in enumerate(labels):
        if lbl == -1:
            continue  # noise
        clusters.setdefault(f"cluster_{lbl}", []).append(idx)

    json.dump(clusters, open(CLUSTER_META_PATH, "w"), indent=2)

    update_last_cluster_time()

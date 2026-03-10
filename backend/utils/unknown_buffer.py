import os, json, time
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from . import config
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from hdbscan import HDBSCAN, all_points_membership_vectors
import umap

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
    image.save(os.path.join(config.IMG_DIR, fname))

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

    # # Trigger 2: time
    # now = time.time()
    # if os.path.exists(config.LAST_CLUSTER_TIME_PATH):
    #     last = float(open(config.LAST_CLUSTER_TIME_PATH).read())
    #     if now - last >= config.CLUSTER_TIME_THRESHOLD:
    #         return True, "TIME_TRIGGER"

    return False, None


# CLUSTERING
def run_clustering():
    if not os.path.exists(config.EMB_PATH):
        return

    embs = np.load(config.EMB_PATH)
    embs = normalize(embs, axis=1) 

    clusterer = HDBSCAN(
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


def run_clustering_HDBSCAN():
    if not os.path.exists(config.EMB_PATH):
        print(f"Error: {config.EMB_PATH} not found.")
        return
    
    # 1. Load and Preprocess
    embs = np.load(config.EMB_PATH)
    # Standardize features before UMAP for better spatial density estimation
    embs_scaled = StandardScaler().fit_transform(embs)
    
    # 2. Advanced Dimensionality Reduction (UMAP)
    # Using 'densmap=True' helps preserve local density which is vital for HDBSCAN
    reducer = umap.UMAP(
        n_components=30,       # Reduced slightly for better cluster density
        n_neighbors=20,       # Increased to capture more global spatial context
        min_dist=0.0,         # Set to 0.0 for better clustering performance
        metric='cosine',
        densmap=True,         # Better preserves relative density
        random_state=42
    )
    embs_reduced = reducer.fit_transform(embs_scaled)
    
    # 3. Enhanced HDBSCAN Clustering
    clusterer = HDBSCAN(
        min_cluster_size=5,
        min_samples=2,         # Lower min_samples reduces noise for tighter spatial grouping
        metric='euclidean',
        cluster_selection_method='leaf', # 'leaf' often yields finer-grained spatial clusters than 'eom'
        prediction_data=True
    )
    labels = clusterer.fit_predict(embs_reduced)
    
    # 4. Spatial Refinement: Soft Clustering
    # This assigns a probability score to every point for every cluster
    soft_clusters = all_points_membership_vectors(clusterer)
    
    # 5. Metrics & Evaluation
    mask = labels != -1
    if mask.sum() >= 2:
        try:
            n_clusters = len(np.unique(labels[mask]))
            if n_clusters >= 2:
                s_score = silhouette_score(embs_reduced[mask], labels[mask])
                db_index = davies_bouldin_score(embs_reduced[mask], labels[mask])
                print(f"--- Spatial Metrics ---")
                print(f"Clusters: {n_clusters} | Noise: {np.sum(labels == -1)}")
                print(f"Silhouette: {s_score:.3f} | DB Index: {db_index:.3f}")
        except Exception as e:
            print(f"Evaluation error: {e}")

    # 6. Structured Output Generation
    clusters = {}
    cluster_info = {}
    
    for idx, lbl in enumerate(labels):
        if lbl == -1:
            # Optional: Use soft clustering to assign noise to the "closest" spatial cluster 
            # if the probability is high enough (> 0.5)
            best_pick = np.argmax(soft_clusters[idx])
            if soft_clusters[idx][best_pick] > 0.7:
                lbl = best_pick
            else:
                continue 
        
        cluster_key = f"cluster_{lbl}"
        if cluster_key not in clusters:
            clusters[cluster_key] = []
            cluster_info[cluster_key] = {'size': 0, 'confidence_scores': []}
        
        clusters[cluster_key].append(int(idx))
        cluster_info[cluster_key]['size'] += 1
        cluster_info[cluster_key]['confidence_scores'].append(float(clusterer.probabilities_[idx]))
    
    # Calculate final cluster health
    for key in cluster_info:
        scores = cluster_info[key]['confidence_scores']
        cluster_info[key]['avg_confidence'] = float(np.mean(scores)) if scores else 0
        del cluster_info[key]['confidence_scores'] # Clean up raw lists

    output = {
        'metadata': {'total_samples': len(embs), 'reduction_method': 'UMAP+DensMAP'},
        'clusters': clusters,
        'cluster_info': cluster_info 
    }
    
    with open(config.CLUSTER_META_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Spatial clustering complete. Saved to {config.CLUSTER_META_PATH}")

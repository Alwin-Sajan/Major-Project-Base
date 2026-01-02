
# ---- Trigger thresholds ----
UNKNOWN_COUNT_THRESHOLD = 4       # Trigger 1
CLUSTER_TIME_THRESHOLD = 24 * 3600  # Trigger 2 (24 hrs)

# ---- Paths ----
UNKNOWN_DIR = "unknown_buffer"
IMG_DIR = f"{UNKNOWN_DIR}/images"
EMB_PATH = f"{UNKNOWN_DIR}/embeddings.npy"
META_PATH = f"{UNKNOWN_DIR}/metadata.json"
CLUSTER_META_PATH = f"{UNKNOWN_DIR}/clusters.json"
LAST_CLUSTER_TIME_PATH = f"{UNKNOWN_DIR}/last_cluster_time.txt"

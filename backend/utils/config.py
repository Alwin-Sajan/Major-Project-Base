# ---- DETECTION MODEL------
CONVNEXT_MODEL_PATH = "/home/abk/abk/projects/Major-project-basic-ui/models/convnext_5epoch.pth"


# ---- Trigger thresholds ----
UNKNOWN_COUNT_THRESHOLD = 4       # Trigger 1
CLUSTER_TIME_THRESHOLD = 24 * 3600  # Trigger 2 (24 hrs)
OOD_THRESHOLD = 0.65     # cosine similarity
MARGIN_THRESHOLD = 0.15 # top1 - top2 gap


# ---- Paths ----
UNKNOWN_DIR = "unknown_buffer"
IMG_DIR = f"{UNKNOWN_DIR}/images"
EMB_PATH = f"{UNKNOWN_DIR}/embeddings.npy"
META_PATH = f"{UNKNOWN_DIR}/metadata.json"
CLUSTER_META_PATH = f"{UNKNOWN_DIR}/clusters.json"
LAST_CLUSTER_TIME_PATH = f"{UNKNOWN_DIR}/last_cluster_time.txt"
CLUSTER_INCREMENTAL_LEARNING_PATH = r"/media/abk/New Disk/DATASETS/CLUSTER_INCREMENTAL_LEARNING"
TRIAL_IMG_DIR = r"/media/abk/New Disk/DATASETS/clusterdataset"


DB_FAISS_PATH = r"/home/abk/abk/projects/Major-project-basic-ui/backend/vectorstore/"
CHAT_TYPE_DETECTION = r"/home/abk/abk/projects/Major-project-basic-ui/backend/vectorstore/chat_type_detection_embed.npz"


# ----- Models ------
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
EMBEDDING_BGE_LARGE = "BAAI/bge-large-en-v1.5"
EMBEDDING_E5_LARGE = "intfloat/e5-large-v2"
EMBEDDING_NVIDIA_LLAMA_NEMOTRON = "nvidia/llama-nemotron-embed-1b-v2"
EMBEDDING_MXBAI = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_NOMIC_TEXT_V2_MOE = "nomic-ai/nomic-embed-text-v2-moe"
EMBEDDING_SNOWFLAKE_ARTIC_LARGE = "Snowflake/snowflake-arctic-embed-l-v2.0"
EMBEDDING_SNOWFLAKE_ARTIC_MEDIUM = "Snowflake/snowflake-arctic-embed-m-v2.0"
EMBEDDING_BGE_M3 = "BAAI/bge-m3"

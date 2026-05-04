# ---- DETECTION MODEL------
CONVNEXT_MODEL_PATH = r"D:\ASP\Major Project\Project\Major-Project-Base\models\convnext_5epoch.pth"


# ---- Trigger thresholds ----
UNKNOWN_COUNT_THRESHOLD = 200       # Trigger 1
CLUSTER_TIME_THRESHOLD = 24 * 3600  # Trigger 2 (24 hrs)
OOD_THRESHOLD = 0.65     # cosine similaritypp
MARGIN_THRESHOLD = 0.15  # top1 - top2 gap


# ---- Paths ----
UNKNOWN_DIR = r"D:\ASP\Major Project\Project\Major-Project-Base\unknown_buffer"
IMG_DIR = rf"{UNKNOWN_DIR}\images"
EMB_PATH = rf"{UNKNOWN_DIR}\embeddings.npy"
META_PATH = rf"{UNKNOWN_DIR}\metadata.json"
CLUSTER_META_PATH = rf"{UNKNOWN_DIR}\clusters.json"
LAST_CLUSTER_TIME_PATH = rf"{UNKNOWN_DIR}\last_cluster_time.txt"

JSONL_RAG_PATH = r"D:\ASP\Major Project\Project\Major-Project-Base\backend\taxonomy_data\merged_taxonomic_chunks.jsonl"

CLUSTER_INCREMENTAL_LEARNING_PATH = r"D:\ASP\Major Project\Project\Major-Project-Base\DATASETS\CLUSTER_INCREMENTAL_LEARNING"
TRIAL_IMG_DIR = r"D:\ASP\Major Project\Project\Major-Project-Base\DATASETS\clusterdataset"


DB_FAISS_PATH = r"D:\ASP\Major Project\Project\Major-Project-Base\backend\vectorstore"
CHAT_TYPE_DETECTION = r"D:\ASP\Major Project\Project\Major-Project-Base\backend\vectorstore\chat_type_detection_embed.npz"


# ----- Models ------
MODEL_LLAMA = "llama3.1:8b-instruct-q4_K_M"
EMBEDDING_BGE_LARGE = "BAAI/bge-large-en-v1.5"
EMBEDDING_E5_LARGE = "intfloat/e5-large-v2"
EMBEDDING_NVIDIA_LLAMA_NEMOTRON = "nvidia/llama-nemotron-embed-1b-v2"
EMBEDDING_MXBAI = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_NOMIC_TEXT_V2_MOE = "nomic-ai/nomic-embed-text-v2-moe"
EMBEDDING_SNOWFLAKE_ARTIC_LARGE = "Snowflake/snowflake-arctic-embed-l-v2.0"
EMBEDDING_SNOWFLAKE_ARTIC_MEDIUM = "Snowflake/snowflake-arctic-embed-m-v2.0"
EMBEDDING_BGE_M3 = "BAAI/bge-m3"
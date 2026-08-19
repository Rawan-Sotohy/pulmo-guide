# ============================================================
# Pulmo Guide - Final Retrieval Configuration
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Hybrid Retrieval
ALPHA = 0.70
SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30

# Number of chunks returned to the LLM
FINAL_TOP_K = 5

# No reranking
USE_RERANKER = False
RERANKER_MODEL_NAME = None
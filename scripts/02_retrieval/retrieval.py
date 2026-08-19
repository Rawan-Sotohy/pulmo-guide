import chromadb
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

import numpy as np


# ============================================================
# PULMO GUIDE
# DAY 2 - FINAL RETRIEVAL + RERANKING
#
# Stage 1:
#   Semantic Search
#   +
#   BM25
#   +
#   Normalization
#   +
#   Hybrid Search
#
# Final Decision:
#   Alpha = 0.7
#   Semantic = 70%
#   BM25 = 30%
#
# Stage 2:
#   MS-MARCO Cross-Encoder Reranking
#
# Pipeline:
#
# Query
#   ↓
# Hybrid Retrieval
#   ↓
# Top 10 Candidates
#   ↓
# MS-MARCO Cross-Encoder
#   ↓
# Reranking
#   ↓
# Final Top 5
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_DB_DIR = BASE_DIR / "data" / "vector_store"

COLLECTION_NAME = "pulmo_guide"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# FINAL DECISION FROM OUR EXPERIMENTS
ALPHA = 0.7

# Retrieve more candidates before reranking
CANDIDATE_K = 10

# Final number of results
FINAL_TOP_K = 5


# ============================================================
# TEST QUERY
# ============================================================

QUERY = (
    "What imaging should be offered to people with stage 3 NSCLC "
    "who are having treatment with curative intent?"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PULMO GUIDE")
print("DAY 2 - FINAL HYBRID RETRIEVAL + MS-MARCO RERANKING")
print("=" * 70)

print("\nFinal Retrieval Configuration:")
print(f"Semantic Weight : {ALPHA}")
print(f"BM25 Weight     : {1 - ALPHA}")
print(f"Candidate K     : {CANDIDATE_K}")
print(f"Final Top K     : {FINAL_TOP_K}")


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("OK: Embedding model loaded.")


# ============================================================
# 2. LOAD MS-MARCO CROSS-ENCODER
# ============================================================

print("\nLoading MS-MARCO Cross-Encoder reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)

print("OK: MS-MARCO Cross-Encoder loaded.")


# ============================================================
# 3. CONNECT TO CHROMADB
# ============================================================

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("OK: Collection loaded.")
print(f"Total chunks in database: {collection.count()}")


# ============================================================
# 4. LOAD ALL CHUNKS
# ============================================================

print("\nLoading all chunks...")

all_data = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

chunk_ids = all_data["ids"]
chunk_texts = all_data["documents"]
chunk_metadata = all_data["metadatas"]

print(
    f"OK: Loaded {len(chunk_texts)} chunks."
)


# ============================================================
# 5. BUILD BM25 INDEX
# ============================================================

print("\nBuilding BM25 index...")

tokenized_documents = [
    text.lower().split()
    for text in chunk_texts
]

bm25 = BM25Okapi(
    tokenized_documents
)

print("OK: BM25 index created.")


# ============================================================
# 6. SCORE NORMALIZATION
# ============================================================

def min_max_normalize(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# 7. HYBRID RETRIEVAL
# ============================================================

def hybrid_search(
    query,
    candidate_k=10,
    alpha=0.7
):

    # --------------------------------------------------------
    # Semantic scores
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    semantic_results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=len(chunk_texts),
        include=[
            "distances"
        ]
    )

    semantic_ids = semantic_results["ids"][0]
    semantic_distances = semantic_results["distances"][0]

    semantic_score_map = {}

    for chunk_id, distance in zip(
        semantic_ids,
        semantic_distances
    ):

        semantic_score_map[chunk_id] = (
            1 - distance
        )


    # --------------------------------------------------------
    # BM25 scores
    # --------------------------------------------------------

    tokens = query.lower().split()

    bm25_scores = bm25.get_scores(
        tokens
    )

    bm25_score_map = {
        chunk_id: float(score)
        for chunk_id, score in zip(
            chunk_ids,
            bm25_scores
        )
    }


    # --------------------------------------------------------
    # Align scores
    # --------------------------------------------------------

    semantic_scores = np.array([
        semantic_score_map[chunk_id]
        for chunk_id in chunk_ids
    ])

    keyword_scores = np.array([
        bm25_score_map[chunk_id]
        for chunk_id in chunk_ids
    ])


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    semantic_normalized = min_max_normalize(
        semantic_scores
    )

    keyword_normalized = min_max_normalize(
        keyword_scores
    )


    # --------------------------------------------------------
    # Hybrid
    #
    # Hybrid =
    # 0.7 Semantic
    # +
    # 0.3 BM25
    # --------------------------------------------------------

    hybrid_scores = (
        alpha * semantic_normalized
        +
        (1 - alpha) * keyword_normalized
    )


    # --------------------------------------------------------
    # Get candidates
    # --------------------------------------------------------

    indices = np.argsort(
        hybrid_scores
    )[::-1][:candidate_k]

    candidates = []

    for rank, index in enumerate(
        indices,
        start=1
    ):

        candidates.append({

            "hybrid_rank": rank,

            "chunk_id": chunk_ids[index],

            "text": chunk_texts[index],

            "metadata": chunk_metadata[index],

            "semantic_score":
                float(semantic_scores[index]),

            "semantic_normalized":
                float(
                    semantic_normalized[index]
                ),

            "bm25_score":
                float(keyword_scores[index]),

            "bm25_normalized":
                float(
                    keyword_normalized[index]
                ),

            "hybrid_score":
                float(hybrid_scores[index])
        })


    return candidates


# ============================================================
# 8. MS-MARCO RERANKING
# ============================================================

def rerank_candidates(
    query,
    candidates,
    final_top_k=5
):

    print(
        "\nRunning MS-MARCO Cross-Encoder reranking..."
    )

    # --------------------------------------------------------
    # Create query-passage pairs
    # --------------------------------------------------------

    pairs = [
        (
            query,
            candidate["text"]
        )
        for candidate in candidates
    ]


    # --------------------------------------------------------
    # Predict relevance scores
    #
    # MS-MARCO returns one relevance score
    # for each query-passage pair.
    # --------------------------------------------------------

    scores = reranker.predict(
        pairs
    )

    scores = np.asarray(
        scores,
        dtype=float
    )


    # --------------------------------------------------------
    # Add reranker score
    # --------------------------------------------------------

    for candidate, score in zip(
        candidates,
        scores
    ):

        candidate["rerank_score"] = float(
            score
        )


    # --------------------------------------------------------
    # Sort by relevance score
    # --------------------------------------------------------

    reranked = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Assign final rank
    # --------------------------------------------------------

    final_results = []

    for rank, candidate in enumerate(
        reranked[:final_top_k],
        start=1
    ):

        candidate["final_rank"] = rank

        final_results.append(
            candidate
        )


    return final_results


# ============================================================
# 9. PRINT HYBRID CANDIDATES
# ============================================================

def print_hybrid_candidates(
    candidates
):

    print("\n" + "=" * 70)
    print("HYBRID CANDIDATES BEFORE RERANKING")
    print("=" * 70)

    for candidate in candidates:

        metadata = candidate["metadata"]

        print("\n" + "-" * 70)

        print(
            f"Hybrid Rank : "
            f"{candidate['hybrid_rank']}"
        )

        print(
            f"Chunk ID    : "
            f"{candidate['chunk_id']}"
        )

        print(
            f"Semantic N  : "
            f"{candidate['semantic_normalized']:.4f}"
        )

        print(
            f"BM25 N      : "
            f"{candidate['bm25_normalized']:.4f}"
        )

        print(
            f"Hybrid      : "
            f"{candidate['hybrid_score']:.4f}"
        )

        print(
            f"Section     : "
            f"{metadata.get('section', '')}"
        )

        print(
            f"Pages       : "
            f"{metadata.get('page_start', '')}"
            f"-"
            f"{metadata.get('page_end', '')}"
        )


# ============================================================
# 10. PRINT FINAL RERANKED RESULTS
# ============================================================

def print_final_results(
    results
):

    print("\n" + "=" * 70)
    print("FINAL MS-MARCO RERANKED RESULTS")
    print("=" * 70)

    for result in results:

        metadata = result["metadata"]

        print("\n" + "-" * 70)

        print(
            f"Final Rank  : "
            f"{result['final_rank']}"
        )

        print(
            f"Chunk ID    : "
            f"{result['chunk_id']}"
        )

        print(
            f"Original Hybrid Rank : "
            f"{result['hybrid_rank']}"
        )

        print(
            f"Hybrid Score : "
            f"{result['hybrid_score']:.4f}"
        )

        print(
            f"Rerank Score : "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"Section      : "
            f"{metadata.get('section', '')}"
        )

        print(
            f"Pages        : "
            f"{metadata.get('page_start', '')}"
            f"-"
            f"{metadata.get('page_end', '')}"
        )

        print(
            f"Citation     : "
            f"{metadata.get('citation', '')}"
        )

        print("\nText:")
        print(
            result["text"]
        )


# ============================================================
# 11. RUN PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

print(QUERY)


# ------------------------------------------------------------
# Stage 1: Hybrid Retrieval
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 1 - HYBRID RETRIEVAL")
print("=" * 70)

hybrid_candidates = hybrid_search(
    QUERY,
    candidate_k=CANDIDATE_K,
    alpha=ALPHA
)

print_hybrid_candidates(
    hybrid_candidates
)


# ------------------------------------------------------------
# Stage 2: MS-MARCO Reranking
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 2 - MS-MARCO CROSS-ENCODER RERANKING")
print("=" * 70)

final_results = rerank_candidates(
    QUERY,
    hybrid_candidates,
    final_top_k=FINAL_TOP_K
)

print_final_results(
    final_results
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RETRIEVAL PIPELINE COMPLETE")
print("=" * 70)

print("\nPipeline:")

print(
    "Query"
    " -> Semantic Search"
    " + BM25"
    " -> Normalization"
    " -> Hybrid (0.7 / 0.3)"
    " -> Top 10"
    " -> MS-MARCO Cross-Encoder"
    " -> Final Top 5"
)

print("\nFinal configuration:")

print("Alpha = 0.7")
print("Semantic = 70%")
print("BM25 = 30%")
print(f"Candidates before reranking = {CANDIDATE_K}")
print(f"Final results = {FINAL_TOP_K}")

print("\nReranker:")
print(RERANKER_MODEL_NAME)

print("=" * 70)
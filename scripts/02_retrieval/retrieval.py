import chromadb
from pathlib import Path

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import numpy as np


# ============================================================
# PULMO GUIDE
# UNIFIED HYBRID RETRIEVAL
#
# CORE:
#   ChromaDB
#
# PATIENT:
#   Temporary chunks in session cache
#   NO Core ChromaDB insertion
#
# Retrieval:
#   Semantic 70%
#   BM25     30%
#   Reranker OFF
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_DB_DIR = (
    BASE_DIR
    / "data"
    / "vector_store"
)

COLLECTION_NAME = "pulmo_guide"

EMBEDDING_MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
)

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30

FINAL_TOP_K = 5

USE_RERANKER = False


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print(
    "OK: Embedding model loaded."
)


# ============================================================
# CONNECT TO CORE CHROMADB
# ============================================================

print("\nConnecting to Core ChromaDB...")

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("OK: Core collection loaded.")

print(
    f"Core chunks in database: "
    f"{collection.count()}"
)


# ============================================================
# LOAD CORE CHUNKS
# ============================================================

print("\nLoading Core chunks...")

all_data = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

CORE_CHUNK_IDS = all_data["ids"]

CORE_CHUNK_TEXTS = all_data["documents"]

CORE_CHUNK_METADATA = all_data["metadatas"]

print(
    f"OK: Loaded "
    f"{len(CORE_CHUNK_TEXTS)} Core chunks."
)


# ============================================================
# BUILD CORE BM25
# ============================================================

print("\nBuilding Core BM25 index...")

CORE_TOKENIZED_DOCUMENTS = [

    text.lower().split()

    for text in CORE_CHUNK_TEXTS

]

CORE_BM25 = BM25Okapi(
    CORE_TOKENIZED_DOCUMENTS
)

print("OK: Core BM25 index created.")


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def min_max_normalize(scores):
    """
    Normalize scores to [0, 1].
    """

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(scores) == 0:
        return scores

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:

        return np.zeros_like(
            scores
        )

    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query_embedding,
    document_embeddings
):
    """
    Calculate cosine similarity between
    one query embedding and document embeddings.
    """

    query_embedding = np.asarray(
        query_embedding,
        dtype=float
    )

    document_embeddings = np.asarray(
        document_embeddings,
        dtype=float
    )

    query_norm = np.linalg.norm(
        query_embedding
    )

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1
    )

    denominator = (
        query_norm * document_norms
    )

    denominator = np.where(
        denominator == 0,
        1e-10,
        denominator
    )

    return (
        document_embeddings
        @
        query_embedding
    ) / denominator


# ============================================================
# CHROMA L2 → COSINE
# ============================================================

def l2_distance_to_cosine_similarity(
    distance
):
    """
    ChromaDB uses squared L2 distance
    because embeddings are normalized.

    cosine = 1 - L2² / 2
    """

    return 1.0 - (
        float(distance) / 2.0
    )


# ============================================================
# BUILD RESULT
# ============================================================

def build_result(
    rank,
    chunk_id,
    text,
    metadata,
    semantic_score,
    semantic_normalized,
    bm25_score,
    bm25_normalized,
    hybrid_score
):

    return {

        "hybrid_rank":
            rank,

        "chunk_id":
            chunk_id,

        "text":
            text,

        "metadata":
            metadata,

        "semantic_score":
            float(
                semantic_score
            ),

        "semantic_normalized":
            float(
                semantic_normalized
            ),

        "bm25_score":
            float(
                bm25_score
            ),

        "bm25_normalized":
            float(
                bm25_normalized
            ),

        "hybrid_score":
            float(
                hybrid_score
            )
    }


# ============================================================
# CORE HYBRID RETRIEVAL
# ============================================================

def hybrid_search(
    query,
    final_top_k=FINAL_TOP_K,
    alpha=SEMANTIC_WEIGHT
):
    """
    Hybrid retrieval from Core ChromaDB.

    Semantic = alpha
    BM25     = 1 - alpha
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    if not 0 <= alpha <= 1:

        raise ValueError(
            "alpha must be between 0 and 1."
        )

    if not CORE_CHUNK_TEXTS:

        return []

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = (
        embedding_model.encode(
            query,
            normalize_embeddings=True
        )
    )

    # --------------------------------------------------------
    # Semantic Search
    # --------------------------------------------------------

    semantic_results = collection.query(

        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=len(
            CORE_CHUNK_TEXTS
        ),

        include=[
            "distances"
        ]
    )

    semantic_ids = (
        semantic_results["ids"][0]
    )

    semantic_distances = (
        semantic_results["distances"][0]
    )

    semantic_score_map = {}

    for chunk_id, distance in zip(
        semantic_ids,
        semantic_distances
    ):

        semantic_score_map[
            chunk_id
        ] = (
            l2_distance_to_cosine_similarity(
                distance
            )
        )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    tokens = query.lower().split()

    bm25_scores = CORE_BM25.get_scores(
        tokens
    )

    bm25_score_map = {

        chunk_id: float(score)

        for chunk_id, score in zip(
            CORE_CHUNK_IDS,
            bm25_scores
        )
    }

    # --------------------------------------------------------
    # Align scores
    # --------------------------------------------------------

    semantic_scores = np.array([

        semantic_score_map.get(
            chunk_id,
            0.0
        )

        for chunk_id in CORE_CHUNK_IDS

    ])

    keyword_scores = np.array([

        bm25_score_map.get(
            chunk_id,
            0.0
        )

        for chunk_id in CORE_CHUNK_IDS

    ])

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    semantic_normalized = (
        min_max_normalize(
            semantic_scores
        )
    )

    keyword_normalized = (
        min_max_normalize(
            keyword_scores
        )
    )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid_scores = (

        alpha
        * semantic_normalized

        +

        (1 - alpha)
        * keyword_normalized
    )

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    final_top_k = min(
        final_top_k,
        len(CORE_CHUNK_IDS)
    )

    indices = np.argsort(
        hybrid_scores
    )[::-1][
        :final_top_k
    ]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    final_results = []

    for rank, index in enumerate(
        indices,
        start=1
    ):

        final_results.append(

            build_result(

                rank=rank,

                chunk_id=
                    CORE_CHUNK_IDS[index],

                text=
                    CORE_CHUNK_TEXTS[index],

                metadata=
                    CORE_CHUNK_METADATA[index],

                semantic_score=
                    semantic_scores[index],

                semantic_normalized=
                    semantic_normalized[index],

                bm25_score=
                    keyword_scores[index],

                bm25_normalized=
                    keyword_normalized[index],

                hybrid_score=
                    hybrid_scores[index]
            )
        )

    return final_results


# ============================================================
# PATIENT HYBRID RETRIEVAL
# ============================================================

def patient_hybrid_search(
    query,
    chunks,
    final_top_k=FINAL_TOP_K,
    alpha=SEMANTIC_WEIGHT,
    session_id=None
):
    """
    Hybrid retrieval from temporary Patient chunks.

    Patient chunks are NOT stored in Core ChromaDB.

    chunks:
        Patient chunks loaded from session cache.

    Required chunk structure:

        {
            "chunk_id": "...",
            "text": "...",
            "embedding": [...],
            "metadata": {...}
        }
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    if not chunks:

        return []

    if not 0 <= alpha <= 1:

        raise ValueError(
            "alpha must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Prepare Patient data
    # --------------------------------------------------------

    valid_chunks = []

    for chunk in chunks:

        text = chunk.get(
            "text",
            ""
        )

        embedding = chunk.get(
            "embedding"
        )

        if not text.strip():
            continue

        if embedding is None:
            continue

        valid_chunks.append(
            chunk
        )

    if not valid_chunks:

        return []

    # --------------------------------------------------------
    # IDs / Text / Metadata
    # --------------------------------------------------------

    patient_ids = [

        chunk.get(
            "chunk_id",
            f"patient_{index}"
        )

        for index, chunk
        in enumerate(valid_chunks)

    ]

    patient_texts = [

        chunk["text"]

        for chunk in valid_chunks

    ]

    patient_embeddings = np.asarray(

        [
            chunk["embedding"]
            for chunk in valid_chunks
        ],

        dtype=float
    )

    patient_metadata = []

    for chunk in valid_chunks:

        metadata = dict(
            chunk.get(
                "metadata",
                {}
            )
        )

        # Make sure Patient is always identified
        metadata["source_type"] = "patient"

        if session_id is not None:

            metadata["session_id"] = (
                session_id
            )

        patient_metadata.append(
            metadata
        )

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = (
        embedding_model.encode(
            query,
            normalize_embeddings=True
        )
    )

    # --------------------------------------------------------
    # Semantic Search
    # --------------------------------------------------------

    semantic_scores = (
        cosine_similarity(
            query_embedding,
            patient_embeddings
        )
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    tokenized_documents = [

        text.lower().split()

        for text in patient_texts

    ]

    patient_bm25 = BM25Okapi(
        tokenized_documents
    )

    tokens = query.lower().split()

    keyword_scores = np.asarray(

        patient_bm25.get_scores(
            tokens
        ),

        dtype=float
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    semantic_normalized = (
        min_max_normalize(
            semantic_scores
        )
    )

    keyword_normalized = (
        min_max_normalize(
            keyword_scores
        )
    )

    # --------------------------------------------------------
    # Hybrid 70/30
    # --------------------------------------------------------

    hybrid_scores = (

        alpha
        * semantic_normalized

        +

        (1 - alpha)
        * keyword_normalized
    )

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    final_top_k = min(
        final_top_k,
        len(patient_ids)
    )

    indices = np.argsort(
        hybrid_scores
    )[::-1][
        :final_top_k
    ]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    final_results = []

    for rank, index in enumerate(
        indices,
        start=1
    ):

        final_results.append(

            build_result(

                rank=rank,

                chunk_id=
                    patient_ids[index],

                text=
                    patient_texts[index],

                metadata=
                    patient_metadata[index],

                semantic_score=
                    semantic_scores[index],

                semantic_normalized=
                    semantic_normalized[index],

                bm25_score=
                    keyword_scores[index],

                bm25_normalized=
                    keyword_normalized[index],

                hybrid_score=
                    hybrid_scores[index]
            )
        )

    return final_results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_final_results(
    results
):

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL HYBRID RETRIEVAL RESULTS"
    )

    print(
        "=" * 70
    )

    for result in results:

        metadata = (
            result.get(
                "metadata",
                {}
            )
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"Rank        : "
            f"{result['hybrid_rank']}"
        )

        print(
            f"Chunk ID    : "
            f"{result['chunk_id']}"
        )

        print(
            f"Source      : "
            f"{metadata.get('source_type', '')}"
        )

        print(
            f"Session     : "
            f"{metadata.get('session_id', '')}"
        )

        print(
            f"Semantic    : "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"BM25        : "
            f"{result['bm25_score']:.4f}"
        )

        print(
            f"Hybrid      : "
            f"{result['hybrid_score']:.4f}"
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

        print(
            f"Citation    : "
            f"{metadata.get('citation', '')}"
        )

        print(
            "\nText:"
        )

        print(
            result["text"]
        )


# ============================================================
# MANUAL CORE TEST
# ============================================================

QUERY = (
    "What imaging should be offered to people with stage 3 NSCLC "
    "who are having treatment with curative intent?"
)


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "PULMO GUIDE - CORE RETRIEVAL TEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nQuery:"
    )

    print(
        QUERY
    )

    results = hybrid_search(
        query=QUERY,
        final_top_k=FINAL_TOP_K,
        alpha=SEMANTIC_WEIGHT
    )

    print_final_results(
        results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Semantic = "
        f"{SEMANTIC_WEIGHT * 100:.0f}%"
    )

    print(
        f"BM25 = "
        f"{BM25_WEIGHT * 100:.0f}%"
    )

    print(
        f"Top K = "
        f"{FINAL_TOP_K}"
    )

    print(
        f"Reranker = "
        f"{'ON' if USE_RERANKER else 'OFF'}"
    )
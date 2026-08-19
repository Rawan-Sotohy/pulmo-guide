import json
import time
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# PULMO GUIDE
# LATENCY EVALUATION
#
# Measures the latency of the FINAL retrieval configuration:
#
#     Semantic Search 70%
#     BM25             30%
#     Hybrid Retrieval
#     Final Top-K = 5
#
# NO reranking
# NO MS-MARCO
# NO NLI
#
# Latency measured per query:
#
#     Query
#       ↓
#     Embedding
#       ↓
#     Semantic Search
#       ↓
#     BM25 scoring
#       ↓
#     Hybrid 70/30
#       ↓
#     Top-5
#
# Model loading, ChromaDB loading and BM25 index creation
# are NOT included in per-query latency.
# ============================================================


# ============================================================
# CONFIGURATION
# Must match the FINAL retrieval configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_DB_DIR = BASE_DIR / "data" / "vector_store"

COLLECTION_NAME = "pulmo_guide"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

EVAL_SET_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "evaluation_set.json"
)

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "evaluation"
)

RESULTS_JSON_PATH = (
    RESULTS_DIR
    / "latency_results.json"
)

RESULTS_MD_PATH = (
    RESULTS_DIR
    / "latency_report.md"
)


# ============================================================
# FINAL RETRIEVAL CONFIGURATION
# ============================================================

ALPHA = 0.70

SEMANTIC_WEIGHT = 0.70

BM25_WEIGHT = 0.30

FINAL_TOP_K = 5


# ============================================================
# OPTIONAL WARM-UP
#
# The first model inference can be slower because of
# initialization / backend warm-up.
#
# We run one warm-up query before measuring the real queries.
# ============================================================

WARMUP_QUERY = "What are the symptoms of lung cancer?"


# ============================================================
# 1. LOAD EVALUATION SET
# ============================================================

def load_evaluation_set(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:

        raise ValueError(
            f"Evaluation set at {path} "
            "is empty or malformed."
        )

    for i, item in enumerate(data):

        if "question" not in item:

            raise ValueError(
                f"Evaluation item {i} "
                "is missing 'question'."
            )

    return data


# ============================================================
# 2. LOAD PIPELINE COMPONENTS
# ============================================================

def load_pipeline_components():

    print("=" * 70)

    print(
        "Loading embedding model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(
        "OK: Embedding model loaded."
    )

    print(
        "\nConnecting to ChromaDB..."
    )

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"OK: Collection loaded. "
        f"Total chunks: {collection.count()}"
    )

    print(
        "\nLoading all chunks..."
    )

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
        f"OK: Loaded "
        f"{len(chunk_texts)} chunks."
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    tokenized_documents = [

        text.lower().split()

        for text in chunk_texts

    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    print(
        "OK: BM25 index created."
    )

    print("=" * 70)

    return {

        "embedding_model":
            embedding_model,

        "collection":
            collection,

        "chunk_ids":
            chunk_ids,

        "chunk_texts":
            chunk_texts,

        "chunk_metadata":
            chunk_metadata,

        "bm25":
            bm25,

    }


# ============================================================
# 3. SCORE NORMALIZATION
# ============================================================

def min_max_normalize(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

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
# 4. HYBRID 70/30 RETRIEVAL
#
# This is the SAME logic used in the evaluation script.
# ============================================================

def hybrid_search(
    query,
    components,
    alpha=ALPHA,
    final_top_k=FINAL_TOP_K
):

    embedding_model = components[
        "embedding_model"
    ]

    collection = components[
        "collection"
    ]

    chunk_ids = components[
        "chunk_ids"
    ]

    bm25 = components[
        "bm25"
    ]

    # --------------------------------------------------------
    # Semantic Search
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    semantic_results = collection.query(

        query_embeddings=[
            query_embedding.tolist()
        ],

        # We retrieve all chunks so that semantic
        # and BM25 scores can be aligned exactly.
        n_results=len(chunk_ids),

        include=[
            "distances"
        ],
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
        ] = 1 - distance

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    tokens = query.lower().split()

    bm25_scores = bm25.get_scores(
        tokens
    )

    bm25_score_map = {

        chunk_id: float(score)

        for chunk_id, score
        in zip(
            chunk_ids,
            bm25_scores
        )

    }

    # --------------------------------------------------------
    # Align scores
    # --------------------------------------------------------

    semantic_scores = np.array([

        semantic_score_map[
            chunk_id
        ]

        for chunk_id in chunk_ids

    ])

    keyword_scores = np.array([

        bm25_score_map[
            chunk_id
        ]

        for chunk_id in chunk_ids

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
    # Final Top-K
    # --------------------------------------------------------

    indices = np.argsort(
        hybrid_scores
    )[::-1][:final_top_k]

    results = []

    for rank, index in enumerate(
        indices,
        start=1
    ):

        results.append({

            "rank":
                rank,

            "chunk_id":
                chunk_ids[index],

            "hybrid_score":
                float(
                    hybrid_scores[index]
                ),

        })

    return results


# ============================================================
# 5. WARM-UP
# ============================================================

def warm_up(components):

    print(
        "\nRunning warm-up query..."
    )

    _ = hybrid_search(
        WARMUP_QUERY,
        components
    )

    print(
        "OK: Warm-up completed."
    )


# ============================================================
# 6. MEASURE ONE QUERY
# ============================================================

def measure_query_latency(
    query,
    components
):

    start = time.perf_counter()

    results = hybrid_search(
        query,
        components
    )

    end = time.perf_counter()

    latency_seconds = (
        end - start
    )

    latency_ms = (
        latency_seconds * 1000
    )

    return (
        latency_ms,
        results
    )


# ============================================================
# 7. CALCULATE STATISTICS
# ============================================================

def calculate_statistics(
    latencies
):

    latencies = np.asarray(
        latencies,
        dtype=float
    )

    return {

        "count":
            int(len(latencies)),

        "average_ms":
            float(np.mean(latencies)),

        "median_ms":
            float(np.median(latencies)),

        "min_ms":
            float(np.min(latencies)),

        "max_ms":
            float(np.max(latencies)),

        "std_ms":
            float(np.std(latencies)),

    }


# ============================================================
# 8. BUILD MARKDOWN REPORT
# ============================================================

def build_markdown_report(
    statistics,
    per_question_results
):

    lines = []

    lines.append(
        "# Pulmo Guide — Retrieval Latency Report"
    )

    lines.append("")

    lines.append(
        "## Final Retrieval Configuration"
    )

    lines.append("")

    lines.append(
        f"- Embedding model: "
        f"`{EMBEDDING_MODEL_NAME}`"
    )

    lines.append(
        "- Retrieval method: `Hybrid Search`"
    )

    lines.append(
        "- Semantic weight: `70%`"
    )

    lines.append(
        "- BM25 weight: `30%`"
    )

    lines.append(
        f"- Alpha: `{ALPHA}`"
    )

    lines.append(
        f"- Final Top-K: `{FINAL_TOP_K}`"
    )

    lines.append(
        "- Reranking: `None`"
    )

    lines.append(
        "- MS-MARCO: `None`"
    )

    lines.append(
        "- NLI: `None`"
    )

    lines.append("")

    lines.append(
        "## Latency Summary"
    )

    lines.append("")

    lines.append(
        "| Metric | Value |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        f"| Number of queries | "
        f"{statistics['count']} |"
    )

    lines.append(
        f"| Average latency | "
        f"{statistics['average_ms']:.2f} ms |"
    )

    lines.append(
        f"| Median latency | "
        f"{statistics['median_ms']:.2f} ms |"
    )

    lines.append(
        f"| Minimum latency | "
        f"{statistics['min_ms']:.2f} ms |"
    )

    lines.append(
        f"| Maximum latency | "
        f"{statistics['max_ms']:.2f} ms |"
    )

    lines.append(
        f"| Standard deviation | "
        f"{statistics['std_ms']:.2f} ms |"
    )

    lines.append("")

    lines.append(
        "## Per-Question Latency"
    )

    lines.append("")

    lines.append(
        "| # | Question | Latency (ms) |"
    )

    lines.append(
        "|---:|---|---:|"
    )

    for i, record in enumerate(
        per_question_results,
        start=1
    ):

        question = (
            record["question"]
            .replace("|", "\\|")
        )

        lines.append(
            f"| {i} | "
            f"{question} | "
            f"{record['latency_ms']:.2f} |"
        )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    start_total = time.perf_counter()

    print("=" * 70)

    print(
        "PULMO GUIDE - HYBRID 70/30 LATENCY TEST"
    )

    print("=" * 70)

    print(
        f"Embedding Model : "
        f"{EMBEDDING_MODEL_NAME}"
    )

    print(
        "Retrieval       : Hybrid Search"
    )

    print(
        "Semantic Weight : 70%"
    )

    print(
        "BM25 Weight     : 30%"
    )

    print(
        f"Final Top-K     : "
        f"{FINAL_TOP_K}"
    )

    print(
        "Reranking       : None"
    )

    print(
        "MS-MARCO        : None"
    )

    print(
        "NLI             : None"
    )

    print(
        f"Evaluation Set  : "
        f"{EVAL_SET_PATH}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load evaluation set
    # --------------------------------------------------------

    print(
        "\nLoading evaluation set..."
    )

    eval_set = load_evaluation_set(
        EVAL_SET_PATH
    )

    print(
        f"OK: Loaded "
        f"{len(eval_set)} questions."
    )

    # --------------------------------------------------------
    # Load pipeline
    # --------------------------------------------------------

    components = (
        load_pipeline_components()
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warm_up(
        components
    )

    # --------------------------------------------------------
    # Measure latency
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MEASURING QUERY LATENCY"
    )

    print(
        "=" * 70
    )

    per_question_results = []

    latencies = []

    for i, item in enumerate(
        eval_set,
        start=1
    ):

        query = item[
            "question"
        ]

        latency_ms, results = (
            measure_query_latency(
                query,
                components
            )
        )

        latencies.append(
            latency_ms
        )

        per_question_results.append({

            "question":
                query,

            "latency_ms":
                latency_ms,

            "retrieved_chunk_ids": [

                result["chunk_id"]

                for result in results

            ],

        })

        print(
            f"[{i}/{len(eval_set)}] "
            f"{latency_ms:.2f} ms | "
            f"{query[:60]}..."
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = (
        calculate_statistics(
            latencies
        )
    )

    total_elapsed = (
        time.perf_counter()
        - start_total
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "LATENCY RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Number of queries : "
        f"{statistics['count']}"
    )

    print(
        f"Average latency  : "
        f"{statistics['average_ms']:.2f} ms"
    )

    print(
        f"Median latency   : "
        f"{statistics['median_ms']:.2f} ms"
    )

    print(
        f"Minimum latency  : "
        f"{statistics['min_ms']:.2f} ms"
    )

    print(
        f"Maximum latency  : "
        f"{statistics['max_ms']:.2f} ms"
    )

    print(
        f"Std deviation    : "
        f"{statistics['std_ms']:.2f} ms"
    )

    print(
        f"\nTotal test time  : "
        f"{total_elapsed:.2f} s"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_payload = {

        "configuration": {

            "embedding_model":
                EMBEDDING_MODEL_NAME,

            "retrieval_method":
                "Hybrid 70/30",

            "alpha":
                ALPHA,

            "semantic_weight":
                SEMANTIC_WEIGHT,

            "bm25_weight":
                BM25_WEIGHT,

            "final_top_k":
                FINAL_TOP_K,

            "reranking":
                False,

            "reranker":
                None,

            "ms_marco":
                False,

            "nli":
                False,

        },

        "statistics":
            statistics,

        "per_question":
            per_question_results,

        "total_test_time_seconds":
            total_elapsed,

    }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    with open(
        RESULTS_JSON_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results_payload,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved JSON results to:"
        f"\n{RESULTS_JSON_PATH}"
    )

    # --------------------------------------------------------
    # Markdown
    # --------------------------------------------------------

    markdown_report = (
        build_markdown_report(
            statistics,
            per_question_results
        )
    )

    with open(
        RESULTS_MD_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            markdown_report
        )

    print(
        f"\nSaved Markdown report to:"
        f"\n{RESULTS_MD_PATH}"
    )

    print(
        "\nDONE."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
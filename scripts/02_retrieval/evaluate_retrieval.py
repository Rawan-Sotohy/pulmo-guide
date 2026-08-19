import argparse
import json
import time
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ============================================================
# PULMO GUIDE
# RETRIEVAL EVALUATION SCRIPT
#
# Compares 4 retrieval configurations against the 32-question
# ground-truth evaluation set:
#
#   1. Semantic Search only
#   2. BM25 only
#   3. Hybrid Search (Alpha = 0.7 / 0.3), no reranking
#   4. Hybrid Search + MS-MARCO Cross-Encoder Reranking
#
# The reranker is trained for Information Retrieval / passage
# ranking, making it suitable for ranking candidate chunks by
# relevance to the query.
#
# This script does NOT modify retrieval.py.
# It reuses the same models, configuration values, hybrid logic,
# and reranking logic used by the production retrieval pipeline.
#
# Metrics reported @5 for every question and averaged overall:
#   Precision@5
#   Recall@5
#   Hit@5
#   MRR@5
# ============================================================


# ============================================================
# CONFIGURATION
# Must match retrieval.py
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_DB_DIR = BASE_DIR / "data" / "vector_store"
COLLECTION_NAME = "pulmo_guide"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# MS-MARCO Cross-Encoder
# Trained for Information Retrieval / passage ranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

ALPHA = 0.7          # 70% Semantic + 30% BM25
CANDIDATE_K = 10     # candidates retrieved before reranking
FINAL_TOP_K = 5      # final results / evaluation cutoff

EVAL_SET_PATH = BASE_DIR / "data" / "evaluation" / "evaluation_set.json"

RESULTS_DIR = BASE_DIR / "data" / "evaluation"

RESULTS_JSON_PATH = (
    RESULTS_DIR / "retrieval_evaluation_results.json"
)

RESULTS_MD_PATH = (
    RESULTS_DIR / "retrieval_evaluation_report.md"
)


CONFIG_LABELS = {
    "semantic": "Semantic Search",
    "bm25": "BM25",
    "hybrid": "Hybrid 70/30",
    "hybrid_rerank": "Hybrid + MS-MARCO",
}


# ============================================================
# ID MAPPING
#
# ChromaDB IDs:
#     core_0040_c8e3ba61
#
# Ground-truth IDs:
#     core_0040
#
# We compare using the base ID.
# ============================================================

def to_base_id(chunk_id: str) -> str:
    parts = chunk_id.split("_")
    return "_".join(parts[:2])


# ============================================================
# 1. LOAD EVALUATION SET
# ============================================================

def load_evaluation_set(path: Path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            f"Evaluation set at {path} is empty or malformed."
        )

    for i, item in enumerate(data):

        if "question" not in item or "relevant_chunk_ids" not in item:
            raise ValueError(
                f"Evaluation item {i} is missing "
                "'question' or 'relevant_chunk_ids': "
                f"{item}"
            )

    return data


# ============================================================
# 2. LOAD MODELS + CHROMADB + BM25
# ============================================================

def load_pipeline_components():

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("OK: Embedding model loaded.")

    print("Loading MS-MARCO Cross-Encoder reranker...")

    reranker = CrossEncoder(
        RERANKER_MODEL_NAME
    )

    print("OK: MS-MARCO Cross-Encoder loaded.")

    print("Connecting to ChromaDB...")

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

    print("Loading all chunks...")

    all_data = collection.get(
        include=["documents", "metadatas"]
    )

    chunk_ids = all_data["ids"]
    chunk_texts = all_data["documents"]
    chunk_metadata = all_data["metadatas"]

    print(
        f"OK: Loaded {len(chunk_texts)} chunks."
    )

    print("Building BM25 index...")

    tokenized_documents = [
        text.lower().split()
        for text in chunk_texts
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    print("OK: BM25 index created.")

    return {
        "embedding_model": embedding_model,
        "reranker": reranker,
        "collection": collection,
        "chunk_ids": chunk_ids,
        "chunk_texts": chunk_texts,
        "chunk_metadata": chunk_metadata,
        "bm25": bm25,
    }


# ============================================================
# 3. SANITY CHECK
# ============================================================

def validate_ground_truth_ids(
    eval_set,
    chunk_ids
):

    known_base_ids = {
        to_base_id(cid)
        for cid in chunk_ids
    }

    missing = {}

    for item in eval_set:

        for gt_id in item["relevant_chunk_ids"]:

            if gt_id not in known_base_ids:

                missing.setdefault(
                    item["question"],
                    []
                ).append(gt_id)

    if missing:

        print(
            "\nWARNING: The following ground-truth IDs "
            "were NOT found"
        )

        print(
            "among the ChromaDB chunk IDs "
            "(after base-ID mapping):"
        )

        for question, ids in missing.items():

            print(
                f"  - '{question[:70]}...' -> {ids}"
            )

        raise ValueError(
            "Ground-truth / ChromaDB ID mismatch detected. "
            "Fix the evaluation set or vector store."
        )

    print(
        f"OK: All ground-truth chunk IDs across "
        f"{len(eval_set)} questions were matched "
        "to ChromaDB base IDs."
    )


# ============================================================
# 4. SCORE NORMALIZATION
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
        / (maximum - minimum)
    )


# ============================================================
# 5. HYBRID RETRIEVAL
#
# Semantic 70%
# BM25     30%
# ============================================================

def hybrid_search(
    query,
    components,
    candidate_k=CANDIDATE_K,
    alpha=ALPHA
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

    chunk_texts = components[
        "chunk_texts"
    ]

    chunk_metadata = components[
        "chunk_metadata"
    ]

    bm25 = components[
        "bm25"
    ]

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
        include=["distances"],
    )

    semantic_ids = semantic_results["ids"][0]

    semantic_distances = (
        semantic_results["distances"][0]
    )

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

            "semantic_score": float(
                semantic_scores[index]
            ),

            "semantic_normalized": float(
                semantic_normalized[index]
            ),

            "bm25_score": float(
                keyword_scores[index]
            ),

            "bm25_normalized": float(
                keyword_normalized[index]
            ),

            "hybrid_score": float(
                hybrid_scores[index]
            ),
        })

    return candidates


# ============================================================
# 6. MS-MARCO RERANKING
#
# The Cross-Encoder receives:
#
#     (query, chunk)
#
# and predicts a relevance score.
#
# Unlike the previous NLI model, this model is designed
# specifically for Information Retrieval / passage ranking.
# ============================================================

def rerank_candidates(
    query,
    candidates,
    reranker,
    final_top_k=FINAL_TOP_K
):

    pairs = [
        (
            query,
            candidate["text"]
        )
        for candidate in candidates
    ]

    # MS-MARCO returns one relevance score per pair
    scores = reranker.predict(
        pairs
    )

    scores = np.asarray(
        scores
    ).reshape(-1)

    for candidate, score in zip(
        candidates,
        scores
    ):

        candidate["rerank_score"] = float(
            score
        )

    # Higher MS-MARCO score = more relevant
    reranked = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

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
# 7. BASELINE HELPERS
# ============================================================

def semantic_only_topk(
    query,
    components,
    k=FINAL_TOP_K
):

    embedding_model = components[
        "embedding_model"
    ]

    collection = components[
        "collection"
    ]

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=k,
        include=["distances"],
    )

    ids = results["ids"][0]

    distances = results["distances"][0]

    return [
        {
            "rank": i + 1,
            "chunk_id": cid,
            "score": 1 - dist,
        }
        for i, (cid, dist)
        in enumerate(
            zip(ids, distances)
        )
    ]


def bm25_only_topk(
    query,
    components,
    k=FINAL_TOP_K
):

    chunk_ids = components[
        "chunk_ids"
    ]

    bm25 = components[
        "bm25"
    ]

    tokens = query.lower().split()

    scores = bm25.get_scores(
        tokens
    )

    order = np.argsort(
        scores
    )[::-1][:k]

    return [
        {
            "rank": i + 1,
            "chunk_id": chunk_ids[idx],
            "score": float(scores[idx]),
        }
        for i, idx in enumerate(order)
    ]


# ============================================================
# 8. METRICS
# ============================================================

def compute_metrics(
    retrieved_chunk_ids_ordered,
    ground_truth_base_ids,
    k=FINAL_TOP_K
):

    gt = set(
        ground_truth_base_ids
    )

    top_k = retrieved_chunk_ids_ordered[
        :k
    ]

    retrieved_base_ids = [
        to_base_id(cid)
        for cid in top_k
    ]

    relevant_flags = [
        base_id in gt
        for base_id in retrieved_base_ids
    ]

    num_relevant_retrieved = sum(
        relevant_flags
    )

    precision_at_k = (
        num_relevant_retrieved / k
        if k
        else 0.0
    )

    recall_at_k = (
        num_relevant_retrieved / len(gt)
        if gt
        else 0.0
    )

    hit_at_k = (
        1
        if num_relevant_retrieved > 0
        else 0
    )

    first_relevant_rank = next(
        (
            i + 1
            for i, flag
            in enumerate(relevant_flags)
            if flag
        ),
        None
    )

    mrr_at_k = (
        1.0 / first_relevant_rank
        if first_relevant_rank
        else 0.0
    )

    return {

        "retrieved_chunk_ids": top_k,

        "retrieved_base_ids": retrieved_base_ids,

        "relevant_flags": relevant_flags,

        "num_relevant_retrieved":
            num_relevant_retrieved,

        "first_relevant_rank":
            first_relevant_rank,

        "precision_at_5":
            precision_at_k,

        "recall_at_5":
            recall_at_k,

        "hit_at_5":
            hit_at_k,

        "mrr_at_5":
            mrr_at_k,
    }


# ============================================================
# 9. EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(
    question_item,
    components,
    k=FINAL_TOP_K
):

    query = question_item[
        "question"
    ]

    ground_truth_ids = question_item[
        "relevant_chunk_ids"
    ]

    # --------------------------------------------------------
    # Config 1: Semantic only
    # --------------------------------------------------------

    semantic_results = semantic_only_topk(
        query,
        components,
        k=k
    )

    semantic_ids_ordered = [
        r["chunk_id"]
        for r in semantic_results
    ]

    # --------------------------------------------------------
    # Config 2: BM25 only
    # --------------------------------------------------------

    bm25_results = bm25_only_topk(
        query,
        components,
        k=k
    )

    bm25_ids_ordered = [
        r["chunk_id"]
        for r in bm25_results
    ]

    # --------------------------------------------------------
    # Config 3 + 4:
    # Same Hybrid candidate pool
    # --------------------------------------------------------

    hybrid_candidates = hybrid_search(
        query,
        components,
        candidate_k=CANDIDATE_K,
        alpha=ALPHA
    )

    # Hybrid without reranking
    hybrid_ids_ordered = [
        c["chunk_id"]
        for c in hybrid_candidates[:k]
    ]

    # Hybrid + MS-MARCO reranking
    reranked = rerank_candidates(
        query,
        list(hybrid_candidates),
        components["reranker"],
        final_top_k=k
    )

    hybrid_rerank_ids_ordered = [
        c["chunk_id"]
        for c in reranked
    ]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    configs_output = {

        "semantic": compute_metrics(
            semantic_ids_ordered,
            ground_truth_ids,
            k=k
        ),

        "bm25": compute_metrics(
            bm25_ids_ordered,
            ground_truth_ids,
            k=k
        ),

        "hybrid": compute_metrics(
            hybrid_ids_ordered,
            ground_truth_ids,
            k=k
        ),

        "hybrid_rerank": compute_metrics(
            hybrid_rerank_ids_ordered,
            ground_truth_ids,
            k=k
        ),
    }

    return {

        "question": query,

        "ground_truth_ids":
            ground_truth_ids,

        "configs":
            configs_output,
    }


# ============================================================
# 10. AGGREGATE METRICS
# ============================================================

def aggregate_metrics(
    per_question_results,
    config_keys
):

    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "hit_at_5",
        "mrr_at_5",
    ]

    averages = {}

    for config_key in config_keys:

        values = {
            m: []
            for m in metric_names
        }

        for record in per_question_results:

            for m in metric_names:

                values[m].append(
                    record[
                        "configs"
                    ][config_key][m]
                )

        averages[config_key] = {

            m: (
                sum(v) / len(v)
                if v
                else 0.0
            )

            for m, v
            in values.items()
        }

    return averages


# ============================================================
# 11. CONSOLE REPORTING
# ============================================================

def print_per_question_result(
    record,
    index,
    total
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"[{index}/{total}] QUESTION: "
        f"{record['question']}"
    )

    print(
        f"Ground truth: "
        f"{record['ground_truth_ids']}"
    )

    for config_key, label in CONFIG_LABELS.items():

        metrics = record[
            "configs"
        ][config_key]

        print(
            f"\n  {label}"
        )

        print(
            f"    Retrieved (top {FINAL_TOP_K}): "
            f"{metrics['retrieved_base_ids']}"
        )

        print(
            f"    Relevant flags: "
            f"{metrics['relevant_flags']}"
        )

        print(
            f"    First relevant rank: "
            f"{metrics['first_relevant_rank']}"
        )

        print(
            f"    Precision@5: "
            f"{metrics['precision_at_5']:.2f}  "
            f"Recall@5: "
            f"{metrics['recall_at_5']:.2f}  "
            f"Hit@5: "
            f"{metrics['hit_at_5']}  "
            f"MRR@5: "
            f"{metrics['mrr_at_5']:.2f}"
        )


# ============================================================
# 12. COMPARISON TABLE
# ============================================================

def build_comparison_table_str(
    averages,
    num_questions
):

    header = (
        f"{'Method':<25}"
        f"{'Precision@5':>14}"
        f"{'Recall@5':>12}"
        f"{'Hit@5':>10}"
        f"{'MRR@5':>10}"
    )

    lines = [
        header,
        "-" * len(header)
    ]

    for config_key, label in CONFIG_LABELS.items():

        a = averages[
            config_key
        ]

        lines.append(
            f"{label:<25}"
            f"{a['precision_at_5']:>14.3f}"
            f"{a['recall_at_5']:>12.3f}"
            f"{a['hit_at_5']:>10.3f}"
            f"{a['mrr_at_5']:>10.3f}"
        )

    lines.append("")

    lines.append(
        f"(Averaged over "
        f"{num_questions} questions, k=5)"
    )

    return "\n".join(lines)


# ============================================================
# 13. MARKDOWN REPORT
# ============================================================

def build_markdown_report(
    per_question_results,
    averages,
    num_questions,
    elapsed_seconds
):

    lines = []

    lines.append(
        "# Pulmo Guide — Retrieval Evaluation Report"
    )

    lines.append("")

    lines.append(
        f"Evaluated **{num_questions} questions** "
        f"from `evaluation_set.json` against "
        f"**{len(CONFIG_LABELS)} retrieval configurations**, "
        f"cutoff k=5, ChromaDB collection "
        f"`{COLLECTION_NAME}`."
    )

    lines.append("")

    lines.append(
        f"Total evaluation time: "
        f"{elapsed_seconds:.1f}s"
    )

    lines.append("")

    lines.append(
        "## Configuration"
    )

    lines.append("")

    lines.append(
        f"- Embedding model: "
        f"`{EMBEDDING_MODEL_NAME}`"
    )

    lines.append(
        f"- Reranker model: "
        f"`{RERANKER_MODEL_NAME}`"
    )

    lines.append(
        f"- Alpha (semantic weight): "
        f"`{ALPHA}`"
    )

    lines.append(
        f"- Semantic weight: `70%`"
    )

    lines.append(
        f"- BM25 weight: `30%`"
    )

    lines.append(
        f"- Candidate K (pre-rerank): "
        f"`{CANDIDATE_K}`"
    )

    lines.append(
        f"- Final Top K: "
        f"`{FINAL_TOP_K}`"
    )

    lines.append("")

    lines.append(
        "## Comparison Table "
        "(averaged over all questions)"
    )

    lines.append("")

    lines.append(
        "| Method | Precision@5 | Recall@5 | "
        "Hit@5 | MRR@5 |"
    )

    lines.append(
        "|---|---|---|---|---|"
    )

    for config_key, label in CONFIG_LABELS.items():

        a = averages[
            config_key
        ]

        lines.append(
            f"| {label} | "
            f"{a['precision_at_5']:.3f} | "
            f"{a['recall_at_5']:.3f} | "
            f"{a['hit_at_5']:.3f} | "
            f"{a['mrr_at_5']:.3f} |"
        )

    lines.append("")

    lines.append(
        "## Per-Question Results"
    )

    lines.append("")

    for i, record in enumerate(
        per_question_results,
        start=1
    ):

        lines.append(
            f"### {i}. "
            f"{record['question']}"
        )

        lines.append("")

        lines.append(
            f"**Ground truth:** "
            f"`{record['ground_truth_ids']}`"
        )

        lines.append("")

        lines.append(
            "| Method | Retrieved (top 5) | "
            "Relevant | First rel. rank | "
            "P@5 | R@5 | Hit@5 | MRR@5 |"
        )

        lines.append(
            "|---|---|---|---|---|---|---|---|"
        )

        for config_key, label in CONFIG_LABELS.items():

            m = record[
                "configs"
            ][config_key]

            lines.append(
                f"| {label} | "
                f"`{m['retrieved_base_ids']}` | "
                f"`{m['relevant_flags']}` | "
                f"{m['first_relevant_rank']} | "
                f"{m['precision_at_5']:.2f} | "
                f"{m['recall_at_5']:.2f} | "
                f"{m['hit_at_5']} | "
                f"{m['mrr_at_5']:.2f} |"
            )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# 14. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 4 retrieval configurations "
            "against evaluation_set.json using "
            "the same production retrieval pipeline."
        )
    )

    parser.add_argument(
        "--eval-set",
        type=Path,
        default=EVAL_SET_PATH,
        help="Path to evaluation_set.json",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only evaluate the first N questions "
            "for quick testing."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress detailed per-question output "
            "and print progress only."
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write result files to disk.",
    )

    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)

    print(
        "PULMO GUIDE - RETRIEVAL EVALUATION"
    )

    print("=" * 70)

    print(
        f"Embedding Model : "
        f"{EMBEDDING_MODEL_NAME}"
    )

    print(
        f"Reranker        : "
        f"{RERANKER_MODEL_NAME}"
    )

    print(
        f"Alpha           : {ALPHA}"
    )

    print(
        f"Semantic Weight : 70%"
    )

    print(
        f"BM25 Weight     : 30%"
    )

    print(
        f"Candidate K     : {CANDIDATE_K}"
    )

    print(
        f"Final Top K     : {FINAL_TOP_K}"
    )

    print(
        f"Evaluation set  : {args.eval_set}"
    )

    # --------------------------------------------------------
    # Load evaluation set
    # --------------------------------------------------------

    eval_set = load_evaluation_set(
        args.eval_set
    )

    if args.limit:
        eval_set = eval_set[
            :args.limit
        ]

    print(
        f"OK: Loaded "
        f"{len(eval_set)} evaluation questions."
    )

    # --------------------------------------------------------
    # Load pipeline
    # --------------------------------------------------------

    components = load_pipeline_components()

    # --------------------------------------------------------
    # Validate ground truth
    # --------------------------------------------------------

    validate_ground_truth_ids(
        eval_set,
        components["chunk_ids"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING 4 RETRIEVAL CONFIGURATIONS "
        "PER QUESTION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    per_question_results = []

    for i, item in enumerate(
        eval_set,
        start=1
    ):

        record = evaluate_question(
            item,
            components,
            k=FINAL_TOP_K
        )

        per_question_results.append(
            record
        )

        if not args.quiet:

            print_per_question_result(
                record,
                i,
                len(eval_set)
            )

        else:

            print(
                f"  [{i}/{len(eval_set)}] "
                f"done: "
                f"{item['question'][:60]}..."
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    config_keys = list(
        CONFIG_LABELS.keys()
    )

    averages = aggregate_metrics(
        per_question_results,
        config_keys
    )

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "COMPARISON TABLE "
        "(averaged over all questions, k=5)"
    )

    print(
        "=" * 70
    )

    print(
        build_comparison_table_str(
            averages,
            len(per_question_results)
        )
    )

    print(
        f"\nTotal evaluation time: "
        f"{elapsed:.1f}s"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if not args.no_save:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        results_payload = {

            "config": {

                "embedding_model":
                    EMBEDDING_MODEL_NAME,

                "reranker_model":
                    RERANKER_MODEL_NAME,

                "alpha":
                    ALPHA,

                "semantic_weight":
                    0.70,

                "bm25_weight":
                    0.30,

                "candidate_k":
                    CANDIDATE_K,

                "final_top_k":
                    FINAL_TOP_K,

                "num_questions":
                    len(per_question_results),
            },

            "averages":
                averages,

            "per_question_results":
                per_question_results,

            "elapsed_seconds":
                elapsed,
        }

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
            f"\nSaved JSON results to: "
            f"{RESULTS_JSON_PATH}"
        )

        markdown_report = (
            build_markdown_report(
                per_question_results,
                averages,
                len(per_question_results),
                elapsed
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
            f"Saved Markdown report to: "
            f"{RESULTS_MD_PATH}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
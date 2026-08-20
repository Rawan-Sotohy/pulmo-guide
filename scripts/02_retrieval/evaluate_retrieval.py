import argparse
import json
import time
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# PULMO GUIDE
# HYBRID RETRIEVAL EVALUATION
#
# Evaluation:
#   Hybrid Retrieval only
#
# Pipeline:
#
# Query
#   ↓
# Semantic Search
#   +
# BM25
#   ↓
# Score Normalization
#   ↓
# Hybrid 70/30
#   ↓
# Top 5
#   ↓
# Evaluation
#
# Metrics:
#   Precision@5
#   Recall@5
#   Hit@5
#   MRR@5
#
# No Reranker
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_DB_DIR = BASE_DIR / "data" / "vector_store"

COLLECTION_NAME = "pulmo_guide"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Hybrid configuration
ALPHA = 0.70

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30

# Final retrieval cutoff
FINAL_TOP_K = 5

# Evaluation set
EVAL_SET_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "evaluation_set.json"
)

# Output files
RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "evaluation"
)

RESULTS_JSON_PATH = (
    RESULTS_DIR
    / "hybrid_retrieval_evaluation_results.json"
)

RESULTS_MD_PATH = (
    RESULTS_DIR
    / "hybrid_retrieval_evaluation_report.md"
)


# ============================================================
# 1. LOAD EVALUATION SET
# ============================================================

def load_evaluation_set(path: Path):

    print("\nLoading evaluation set...")

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):

        raise ValueError(
            "Evaluation set must be a JSON list."
        )

    if len(data) == 0:

        raise ValueError(
            "Evaluation set is empty."
        )

    for i, item in enumerate(data):

        if "question" not in item:

            raise ValueError(
                f"Evaluation item {i} "
                "is missing 'question'."
            )

        if "relevant_chunk_ids" not in item:

            raise ValueError(
                f"Evaluation item {i} "
                "is missing 'relevant_chunk_ids'."
            )

    print(
        f"OK: Loaded {len(data)} evaluation questions."
    )

    return data


# ============================================================
# 2. LOAD MODELS + CHROMADB + BM25
# ============================================================

def load_pipeline_components():

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(
        "OK: Embedding model loaded."
    )

    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        "OK: Collection loaded."
    )

    print(
        f"Total chunks in database: "
        f"{collection.count()}"
    )

    # --------------------------------------------------------
    # Load all chunks
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print("\nBuilding BM25 index...")

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
# 3. ID MAPPING
#
# ChromaDB:
#     core_0040_c8e3ba61
#
# Evaluation:
#     core_0040
#
# We compare using the base ID.
# ============================================================

def to_base_id(chunk_id: str) -> str:

    parts = chunk_id.split("_")

    return "_".join(parts[:2])


# ============================================================
# 4. VALIDATE GROUND-TRUTH IDs
# ============================================================

def validate_ground_truth_ids(
    eval_set,
    chunk_ids
):

    print(
        "\nValidating ground-truth IDs..."
    )

    known_base_ids = {

        to_base_id(cid)

        for cid in chunk_ids

    }

    missing = {}

    for item in eval_set:

        question = item["question"]

        for gt_id in item[
            "relevant_chunk_ids"
        ]:

            if gt_id not in known_base_ids:

                missing.setdefault(
                    question,
                    []
                ).append(gt_id)

    if missing:

        print(
            "\nERROR: Ground-truth IDs "
            "were not found."
        )

        for question, ids in missing.items():

            print(
                f"\nQuestion: {question}"
            )

            print(
                f"Missing IDs: {ids}"
            )

        raise ValueError(
            "Ground-truth / ChromaDB ID mismatch."
        )

    print(
        f"OK: All ground-truth IDs matched "
        f"against {len(chunk_ids)} ChromaDB chunks."
    )


# ============================================================
# 5. SCORE NORMALIZATION
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
# 6. HYBRID SEARCH
#
# Semantic = 70%
# BM25     = 30%
# ============================================================

def hybrid_search(
    query,
    components,
    final_top_k=FINAL_TOP_K,
    alpha=ALPHA
):

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

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

    # ========================================================
    # Semantic Search
    # ========================================================

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

    # ========================================================
    # BM25
    # ========================================================

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

    # ========================================================
    # Align scores
    # ========================================================

    semantic_scores = np.array([

        semantic_score_map[chunk_id]

        for chunk_id in chunk_ids

    ])

    keyword_scores = np.array([

        bm25_score_map[chunk_id]

        for chunk_id in chunk_ids

    ])

    # ========================================================
    # Normalize
    # ========================================================

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

    # ========================================================
    # Hybrid 70 / 30
    # ========================================================

    hybrid_scores = (

        alpha
        * semantic_normalized

        +

        (1 - alpha)
        * keyword_normalized

    )

    # ========================================================
    # Top K
    # ========================================================

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

            "text":
                chunk_texts[index],

            "metadata":
                chunk_metadata[index],

            "semantic_score":
                float(
                    semantic_scores[index]
                ),

            "semantic_normalized":
                float(
                    semantic_normalized[index]
                ),

            "bm25_score":
                float(
                    keyword_scores[index]
                ),

            "bm25_normalized":
                float(
                    keyword_normalized[index]
                ),

            "hybrid_score":
                float(
                    hybrid_scores[index]
                ),

        })

    return results


# ============================================================
# 7. COMPUTE FOUR METRICS
#
# Precision@5
# Recall@5
# Hit@5
# MRR@5
# ============================================================

def compute_metrics(

    retrieved_chunk_ids,

    ground_truth_base_ids,

    k=FINAL_TOP_K

):

    ground_truth = set(
        ground_truth_base_ids
    )

    top_k_ids = retrieved_chunk_ids[
        :k
    ]

    retrieved_base_ids = [

        to_base_id(chunk_id)

        for chunk_id in top_k_ids

    ]

    # --------------------------------------------------------
    # Relevant flags
    # --------------------------------------------------------

    relevant_flags = [

        base_id in ground_truth

        for base_id in retrieved_base_ids

    ]

    num_relevant_retrieved = sum(
        relevant_flags
    )

    # --------------------------------------------------------
    # Precision@5
    #
    # Relevant retrieved / retrieved
    # --------------------------------------------------------

    precision_at_k = (

        num_relevant_retrieved / k

        if k > 0

        else 0.0

    )

    # --------------------------------------------------------
    # Recall@5
    #
    # Relevant retrieved / total relevant
    # --------------------------------------------------------

    recall_at_k = (

        num_relevant_retrieved
        /
        len(ground_truth)

        if ground_truth

        else 0.0

    )

    # --------------------------------------------------------
    # Hit@5
    #
    # At least one relevant chunk?
    # --------------------------------------------------------

    hit_at_k = (

        1

        if num_relevant_retrieved > 0

        else 0

    )

    # --------------------------------------------------------
    # MRR@5
    #
    # 1 / rank of first relevant chunk
    # --------------------------------------------------------

    first_relevant_rank = next(

        (

            i + 1

            for i, flag in enumerate(
                relevant_flags
            )

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

        "retrieved_chunk_ids":
            top_k_ids,

        "retrieved_base_ids":
            retrieved_base_ids,

        "relevant_flags":
            relevant_flags,

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
# 8. EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(
    item,
    components
):

    question = item[
        "question"
    ]

    ground_truth_ids = item[
        "relevant_chunk_ids"
    ]

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    results = hybrid_search(

        question,

        components,

        final_top_k=FINAL_TOP_K,

        alpha=ALPHA

    )

    retrieved_ids = [

        result["chunk_id"]

        for result in results

    ]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = compute_metrics(

        retrieved_ids,

        ground_truth_ids,

        k=FINAL_TOP_K

    )

    return {

        "question":
            question,

        "ground_truth_ids":
            ground_truth_ids,

        "results":
            results,

        "metrics":
            metrics,

    }


# ============================================================
# 9. AGGREGATE METRICS
# ============================================================

def aggregate_metrics(
    per_question_results
):

    metric_names = [

        "precision_at_5",

        "recall_at_5",

        "hit_at_5",

        "mrr_at_5",

    ]

    averages = {}

    for metric in metric_names:

        values = [

            record["metrics"][metric]

            for record
            in per_question_results

        ]

        averages[metric] = (

            sum(values)
            /
            len(values)

            if values

            else 0.0

        )

    return averages


# ============================================================
# 10. PRINT QUESTION RESULT
# ============================================================

def print_question_result(
    record,
    index,
    total
):

    metrics = record[
        "metrics"
    ]

    print(
        "\n" + "-" * 70
    )

    print(
        f"[{index}/{total}] "
        f"{record['question']}"
    )

    print(
        f"\nGround Truth:"
    )

    print(
        record["ground_truth_ids"]
    )

    print(
        "\nRetrieved Top 5:"
    )

    print(
        metrics[
            "retrieved_base_ids"
        ]
    )

    print(
        f"\nRelevant Flags:"
    )

    print(
        metrics[
            "relevant_flags"
        ]
    )

    print(
        f"\nFirst Relevant Rank:"
        f" {metrics['first_relevant_rank']}"
    )

    print(
        f"Precision@5:"
        f" {metrics['precision_at_5']:.3f}"
    )

    print(
        f"Recall@5:"
        f" {metrics['recall_at_5']:.3f}"
    )

    print(
        f"Hit@5:"
        f" {metrics['hit_at_5']}"
    )

    print(
        f"MRR@5:"
        f" {metrics['mrr_at_5']:.3f}"
    )


# ============================================================
# 11. BUILD COMPARISON TABLE
# ============================================================

def build_comparison_table(
    averages,
    num_questions
):

    lines = []

    lines.append(
        "\n" + "=" * 70
    )

    lines.append(
        "HYBRID RETRIEVAL EVALUATION"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Questions evaluated: "
        f"{num_questions}"
    )

    lines.append(
        f"Embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    lines.append(
        f"Semantic weight: "
        f"{SEMANTIC_WEIGHT:.0%}"
    )

    lines.append(
        f"BM25 weight: "
        f"{BM25_WEIGHT:.0%}"
    )

    lines.append(
        f"Final Top K: "
        f"{FINAL_TOP_K}"
    )

    lines.append("")

    lines.append(
        f"{'Metric':<20}"
        f"{'Score':>12}"
    )

    lines.append(
        "-" * 32
    )

    lines.append(
        f"{'Precision@5':<20}"
        f"{averages['precision_at_5']:>12.3f}"
    )

    lines.append(
        f"{'Recall@5':<20}"
        f"{averages['recall_at_5']:>12.3f}"
    )

    lines.append(
        f"{'Hit@5':<20}"
        f"{averages['hit_at_5']:>12.3f}"
    )

    lines.append(
        f"{'MRR@5':<20}"
        f"{averages['mrr_at_5']:>12.3f}"
    )

    lines.append("")

    lines.append(
        "=" * 70
    )

    return "\n".join(lines)


# ============================================================
# 12. BUILD MARKDOWN REPORT
# ============================================================

def build_markdown_report(
    per_question_results,
    averages,
    elapsed_seconds
):

    lines = []

    lines.append(
        "# Pulmo Guide — Hybrid Retrieval Evaluation"
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
        f"- Semantic weight: "
        f"`70%`"
    )

    lines.append(
        f"- BM25 weight: "
        f"`30%`"
    )

    lines.append(
        f"- Final Top K: "
        f"`{FINAL_TOP_K}`"
    )

    lines.append(
        "- Reranker: `OFF`"
    )

    lines.append("")

    lines.append(
        "## Overall Metrics"
    )

    lines.append("")

    lines.append(
        "| Metric | Score |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        f"| Precision@5 | "
        f"{averages['precision_at_5']:.3f} |"
    )

    lines.append(
        f"| Recall@5 | "
        f"{averages['recall_at_5']:.3f} |"
    )

    lines.append(
        f"| Hit@5 | "
        f"{averages['hit_at_5']:.3f} |"
    )

    lines.append(
        f"| MRR@5 | "
        f"{averages['mrr_at_5']:.3f} |"
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

        metrics = record[
            "metrics"
        ]

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
            f"**Retrieved Top 5:** "
            f"`{metrics['retrieved_base_ids']}`"
        )

        lines.append("")

        lines.append(
            f"**Relevant flags:** "
            f"`{metrics['relevant_flags']}`"
        )

        lines.append("")

        lines.append(
            f"- Precision@5: "
            f"`{metrics['precision_at_5']:.3f}`"
        )

        lines.append(
            f"- Recall@5: "
            f"`{metrics['recall_at_5']:.3f}`"
        )

        lines.append(
            f"- Hit@5: "
            f"`{metrics['hit_at_5']}`"
        )

        lines.append(
            f"- MRR@5: "
            f"`{metrics['mrr_at_5']:.3f}`"
        )

        lines.append("")

    lines.append(
        f"Evaluation time: "
        f"{elapsed_seconds:.1f}s"
    )

    return "\n".join(lines)


# ============================================================
# 13. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Pulmo Guide Hybrid "
            "Retrieval using Precision@5, "
            "Recall@5, Hit@5 and MRR@5."
        )
    )

    parser.add_argument(
        "--eval-set",
        type=Path,
        default=EVAL_SET_PATH,
        help="Path to evaluation_set.json"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N "
            "questions for quick testing."
        )
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Show only progress and final "
            "metrics."
        )
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save result files."
    )

    args = parser.parse_args()

    start_time = time.time()

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 70)

    print(
        "PULMO GUIDE"
    )

    print(
        "HYBRID RETRIEVAL EVALUATION"
    )

    print("=" * 70)

    print(
        "\nConfiguration:"
    )

    print(
        f"Embedding Model : "
        f"{EMBEDDING_MODEL_NAME}"
    )

    print(
        f"Semantic Weight : "
        f"{SEMANTIC_WEIGHT:.0%}"
    )

    print(
        f"BM25 Weight     : "
        f"{BM25_WEIGHT:.0%}"
    )

    print(
        f"Final Top K     : "
        f"{FINAL_TOP_K}"
    )

    print(
        "Reranker        : OFF"
    )

    print(
        f"Evaluation Set  : "
        f"{args.eval_set}"
    )

    # ========================================================
    # Load evaluation set
    # ========================================================

    eval_set = load_evaluation_set(
        args.eval_set
    )

    if args.limit is not None:

        eval_set = eval_set[
            :args.limit
        ]

        print(
            f"\nLIMIT MODE: "
            f"Evaluating only "
            f"{len(eval_set)} questions."
        )

    # ========================================================
    # Load pipeline
    # ========================================================

    components = (
        load_pipeline_components()
    )

    # ========================================================
    # Validate IDs
    # ========================================================

    validate_ground_truth_ids(

        eval_set,

        components["chunk_ids"]

    )

    # ========================================================
    # Run evaluation
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING HYBRID 70/30 EVALUATION"
    )

    print(
        "=" * 70
    )

    per_question_results = []

    total_questions = len(
        eval_set
    )

    for i, item in enumerate(
        eval_set,
        start=1
    ):

        record = evaluate_question(

            item,

            components

        )

        per_question_results.append(
            record
        )

        if args.quiet:

            print(
                f"[{i}/{total_questions}] "
                f"done"
            )

        else:

            print_question_result(

                record,

                i,

                total_questions

            )

    # ========================================================
    # Aggregate
    # ========================================================

    averages = aggregate_metrics(

        per_question_results

    )

    elapsed = (
        time.time()
        - start_time
    )

    # ========================================================
    # Final results
    # ========================================================

    print(
        build_comparison_table(

            averages,

            len(per_question_results)

        )
    )

    print(
        f"Evaluation time: "
        f"{elapsed:.1f}s"
    )

    # ========================================================
    # Save results
    # ========================================================

    if not args.no_save:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        results_payload = {

            "config": {

                "embedding_model":
                    EMBEDDING_MODEL_NAME,

                "semantic_weight":
                    SEMANTIC_WEIGHT,

                "bm25_weight":
                    BM25_WEIGHT,

                "alpha":
                    ALPHA,

                "final_top_k":
                    FINAL_TOP_K,

                "reranker":
                    False,

                "num_questions":
                    len(
                        per_question_results
                    ),

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
            f"\nSaved JSON results to:"
            f"\n{RESULTS_JSON_PATH}"
        )

        markdown_report = (
            build_markdown_report(

                per_question_results,

                averages,

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
            f"Saved Markdown report to:"
            f"\n{RESULTS_MD_PATH}"
        )

    # ========================================================
    # DONE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "HYBRID RETRIEVAL EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
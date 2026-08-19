"""
============================================================
PULMO GUIDE
DAY 3 - REFUSAL THRESHOLD TESTING
============================================================

This script does NOT modify:
- retrieval_config.py
- retrieval.py
- Embedding model
- Chunking
- Vector store
- Reranker configuration

It only reuses the final Hybrid Retrieval logic
(Alpha = 0.70, no reranker) to calculate a retrieval
score for each question and evaluate different thresholds
for the Accept / Refuse decision.

Confidence signal:
    hybrid_score of the top-ranked retrieved chunk (Rank 1)

Expected file location:
    scripts/02_retrieval/test_refusal_threshold.py

Input:
    data/evaluation/refusal_evaluation_set.json

Outputs:
    data/evaluation/refusal_threshold_results.json
    data/evaluation/refusal_threshold_report.md
============================================================
"""

import json
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# IMPORT FINAL RETRIEVAL CONFIGURATION
# ============================================================

sys.path.append(
    str(Path(__file__).resolve().parent / "config")
)

from retrieval_config import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    ALPHA,
    FINAL_TOP_K,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_DB_DIR = BASE_DIR / "data" / "vector_store"
COLLECTION_NAME = "pulmo_guide"

EVAL_SET_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "refusal_evaluation_set.json"
)

RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "refusal_threshold_results.json"
)

REPORT_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "refusal_threshold_report.md"
)


# ============================================================
# THRESHOLD VALUES TO TEST
#
# We start around 0.65 as requested for the Day 3 experiment.
# The final threshold will NOT be selected until the results
# are reviewed.
# ============================================================

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


# Same candidate pool used by the retrieval evaluation.
CANDIDATE_K = 10


# ============================================================
# 1. SCORE NORMALIZATION
# ============================================================

def min_max_normalize(scores):
    """
    Normalize scores to the range [0, 1].
    """

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
# 2. BUILD RETRIEVAL INDEXES
# ============================================================

def build_indexes():

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
        include=[
            "documents",
            "metadatas"
        ]
    )

    chunk_ids = all_data["ids"]
    chunk_texts = all_data["documents"]

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

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("OK: Embedding model loaded.")

    return (
        collection,
        chunk_ids,
        chunk_texts,
        bm25,
        embedding_model,
    )


# ============================================================
# 3. HYBRID SEARCH
#
# Same final retrieval configuration:
#
# Semantic = 70%
# BM25     = 30%
# Reranker = None
#
# Returns the highest Hybrid score (Rank 1).
# ============================================================

def hybrid_search_top_score(
    query,
    collection,
    chunk_ids,
    chunk_texts,
    bm25,
    embedding_model,
    candidate_k=CANDIDATE_K,
    alpha=ALPHA,
):
    """
    Run Hybrid Retrieval and return the top-ranked chunk
    together with its Hybrid score.

    The Hybrid score is used only as a retrieval-based
    signal for the refusal experiment.
    """

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
        n_results=len(chunk_texts),
        include=["distances"],
    )

    semantic_ids = semantic_results["ids"][0]

    semantic_distances = (
        semantic_results["distances"][0]
    )

    semantic_score_map = {
        chunk_id: 1 - distance
        for chunk_id, distance
        in zip(
            semantic_ids,
            semantic_distances
        )
    }

    # --------------------------------------------------------
    # BM25 Search
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
    # Align Scores
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
    # Normalize Scores
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
    # Hybrid Score
    #
    # 70% Semantic
    # 30% BM25
    # --------------------------------------------------------

    hybrid_scores = (
        alpha * semantic_normalized
        +
        (1 - alpha) * keyword_normalized
    )

    # --------------------------------------------------------
    # Get Top Candidates
    # --------------------------------------------------------

    indices = np.argsort(
        hybrid_scores
    )[::-1][:candidate_k]

    top_index = indices[0]

    return {
        "top_chunk_id":
            chunk_ids[top_index],

        "hybrid_score":
            float(
                hybrid_scores[top_index]
            ),
    }


# ============================================================
# 4. RUN RETRIEVAL FOR ALL QUESTIONS
# ============================================================

def run_confidence_scoring():

    print(
        "\nLoading evaluation set..."
    )

    with open(
        EVAL_SET_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        eval_set = json.load(f)

    print(
        f"OK: Loaded "
        f"{len(eval_set)} questions."
    )

    (
        collection,
        chunk_ids,
        chunk_texts,
        bm25,
        embedding_model,
    ) = build_indexes()

    scored_items = []

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING HYBRID RETRIEVAL"
    )

    print(
        "=" * 70
    )

    for index, item in enumerate(
        eval_set,
        start=1
    ):

        start_time = time.time()

        result = hybrid_search_top_score(
            query=item["question"],
            collection=collection,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            bm25=bm25,
            embedding_model=embedding_model,
            candidate_k=CANDIDATE_K,
            alpha=ALPHA,
        )

        elapsed_ms = (
            time.time() - start_time
        ) * 1000

        scored_items.append({

            "id":
                item["id"],

            "type":
                item["type"],

            "question":
                item["question"],

            "expected_decision":
                item["expected_decision"],

            "top_chunk_id":
                result["top_chunk_id"],

            "hybrid_score":
                result["hybrid_score"],

            "latency_ms":
                round(
                    elapsed_ms,
                    2
                ),
        })

        print(
            f"[{index}/{len(eval_set)}] "
            f"{item['id']} "
            f"({item['type']}) | "
            f"Score: "
            f"{result['hybrid_score']:.4f}"
        )

    return scored_items


# ============================================================
# 5. EVALUATE THRESHOLDS
# ============================================================

def evaluate_thresholds(
    scored_items,
    thresholds=THRESHOLDS,
):

    results = []

    for threshold in thresholds:

        correct_accept = 0
        correct_refuse = 0

        false_accept = 0
        false_reject = 0

        # ----------------------------------------------------
        # Evaluate every question
        # ----------------------------------------------------

        for item in scored_items:

            decision = (
                "accept"
                if item["hybrid_score"]
                >= threshold
                else "refuse"
            )

            expected = (
                item["expected_decision"]
            )

            if (
                decision == "accept"
                and expected == "accept"
            ):

                correct_accept += 1

            elif (
                decision == "refuse"
                and expected == "refuse"
            ):

                correct_refuse += 1

            elif (
                decision == "accept"
                and expected == "refuse"
            ):

                false_accept += 1

            elif (
                decision == "refuse"
                and expected == "accept"
            ):

                false_reject += 1

        # ----------------------------------------------------
        # Totals
        # ----------------------------------------------------

        accepted = (
            correct_accept
            + false_accept
        )

        rejected = (
            correct_refuse
            + false_reject
        )

        # ----------------------------------------------------
        # Precision
        #
        # Of all accepted questions,
        # how many should actually be accepted?
        # ----------------------------------------------------

        precision = (
            correct_accept / accepted
            if accepted > 0
            else None
        )

        # ----------------------------------------------------
        # Recall
        #
        # Of all questions that should be accepted,
        # how many did we correctly accept?
        # ----------------------------------------------------

        recall = (
            correct_accept
            / (
                correct_accept
                + false_reject
            )
            if (
                correct_accept
                + false_reject
            ) > 0
            else None
        )

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        total = len(scored_items)

        accuracy = (
            (
                correct_accept
                + correct_refuse
            )
            / total
            if total > 0
            else 0.0
        )

        results.append({

            "threshold":
                threshold,

            "accepted":
                accepted,

            "rejected":
                rejected,

            "correct_accept":
                correct_accept,

            "correct_refuse":
                correct_refuse,

            "false_accept":
                false_accept,

            "false_reject":
                false_reject,

            "precision":
                (
                    round(
                        precision,
                        4
                    )
                    if precision is not None
                    else None
                ),

            "recall":
                (
                    round(
                        recall,
                        4
                    )
                    if recall is not None
                    else None
                ),

            "accuracy":
                round(
                    accuracy,
                    4
                ),
        })

    return results


# ============================================================
# 6. SAVE JSON RESULTS
# ============================================================

def save_json(
    scored_items,
    threshold_results,
):

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    payload = {

        "config": {

            "embedding_model":
                EMBEDDING_MODEL_NAME,

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

            "reranker":
                None,

            "confidence_signal":
                "hybrid_score of rank-1 chunk",
        },

        "scored_items":
            scored_items,

        "threshold_results":
            threshold_results,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nOK: Results saved to:\n"
        f"{RESULTS_PATH}"
    )


# ============================================================
# 7. SAVE MARKDOWN REPORT
# ============================================================

def save_markdown_report(
    scored_items,
    threshold_results,
):

    in_kb_count = sum(
        1
        for item in scored_items
        if item["type"] == "in_kb"
    )

    out_of_kb_count = sum(
        1
        for item in scored_items
        if item["type"] == "out_of_kb"
    )

    lines = []

    lines.append(
        "# Pulmo Guide - Refusal Threshold Experiment"
    )

    lines.append("")

    lines.append(
        f"Dataset size: **{len(scored_items)} questions**"
    )

    lines.append(
        f"- In-KB: **{in_kb_count}**"
    )

    lines.append(
        f"- Out-of-KB: **{out_of_kb_count}**"
    )

    lines.append("")

    lines.append(
        "## Retrieval Configuration"
    )

    lines.append("")

    lines.append(
        f"- Embedding model: "
        f"`{EMBEDDING_MODEL_NAME}`"
    )

    lines.append(
        f"- Semantic weight: **70%**"
    )

    lines.append(
        f"- BM25 weight: **30%**"
    )

    lines.append(
        f"- Alpha: `{ALPHA}`"
    )

    lines.append(
        f"- Final Top-K: `{FINAL_TOP_K}`"
    )

    lines.append(
        "- Reranker: **None**"
    )

    lines.append(
        "- Confidence signal: "
        "**Rank-1 Hybrid score**"
    )

    lines.append("")

    lines.append(
        "## Threshold Results"
    )

    lines.append("")

    lines.append(
        "| Threshold | Accepted | Rejected | "
        "Correct Accept | Correct Refuse | "
        "False Accept | False Reject | "
        "Precision | Recall | Accuracy |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---:|"
    )

    for result in threshold_results:

        lines.append(
            f"| {result['threshold']} | "
            f"{result['accepted']} | "
            f"{result['rejected']} | "
            f"{result['correct_accept']} | "
            f"{result['correct_refuse']} | "
            f"{result['false_accept']} | "
            f"{result['false_reject']} | "
            f"{result['precision']} | "
            f"{result['recall']} | "
            f"{result['accuracy']} |"
        )

    lines.append("")

    lines.append(
        "## Interpretation"
    )

    lines.append("")

    lines.append(
        "The Hybrid score is used as a retrieval-based "
        "signal for the refusal experiment. It should not "
        "be interpreted as a calibrated probability."
    )

    lines.append("")

    lines.append(
        "The final threshold should be selected based on "
        "the trade-off between false accepts and false "
        "rejects, with special attention to out-of-KB "
        "questions."
    )

    lines.append("")

    lines.append(
        "**Important:** This experiment is directional. "
        "A larger and more diverse labeled dataset is "
        "recommended before treating the threshold as "
        "fully validated."
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print(
        f"OK: Report saved to:\n"
        f"{REPORT_PATH}"
    )


# ============================================================
# 8. PRINT RESULTS
# ============================================================

def print_results(
    threshold_results,
):

    print(
        "\n" + "=" * 100
    )

    print(
        "THRESHOLD ANALYSIS"
    )

    print(
        "=" * 100
    )

    header = (
        f"{'Threshold':<12}"
        f"{'Accept':<10}"
        f"{'Reject':<10}"
        f"{'Correct':<10}"
        f"{'False Accept':<15}"
        f"{'False Reject':<15}"
        f"{'Precision':<12}"
        f"{'Recall':<10}"
        f"{'Accuracy':<10}"
    )

    print(header)
    print("-" * 100)

    for result in threshold_results:

        print(
            f"{result['threshold']:<12.2f}"
            f"{result['accepted']:<10}"
            f"{result['rejected']:<10}"
            f"{result['correct_accept']:<10}"
            f"{result['false_accept']:<15}"
            f"{result['false_reject']:<15}"
            f"{str(result['precision']):<12}"
            f"{str(result['recall']):<10}"
            f"{result['accuracy']:<10.4f}"
        )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PULMO GUIDE - DAY 3 - "
        "REFUSAL THRESHOLD TESTING"
    )

    print("=" * 70)

    print(
        f"Embedding Model : "
        f"{EMBEDDING_MODEL_NAME}"
    )

    print(
        "Retrieval       : Hybrid"
    )

    print(
        f"Semantic Weight : "
        f"{ALPHA * 100:.0f}%"
    )

    print(
        f"BM25 Weight     : "
        f"{(1 - ALPHA) * 100:.0f}%"
    )

    print(
        f"Final Top-K     : "
        f"{FINAL_TOP_K}"
    )

    print(
        "Reranking       : None"
    )

    print(
        f"Evaluation Set  : "
        f"{EVAL_SET_PATH}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Run retrieval
    # --------------------------------------------------------

    scored_items = (
        run_confidence_scoring()
    )

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    threshold_results = (
        evaluate_thresholds(
            scored_items
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_results(
        threshold_results
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_json(
        scored_items,
        threshold_results
    )

    save_markdown_report(
        scored_items,
        threshold_results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "REFUSAL THRESHOLD EXPERIMENT COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nDo NOT change retrieval_config.py yet."
    )

    print(
        "Review the threshold results first, "
        "then select the final threshold."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
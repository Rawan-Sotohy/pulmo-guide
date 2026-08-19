import json
import sys
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# PULMO GUIDE
# THRESHOLD EXPERIMENT
#
# Purpose:
# Find a suitable confidence threshold for the final
# Hybrid 70/30 retrieval system.
#
# This experiment DOES NOT change production retrieval.
#
# Final retrieval configuration:
#   Embedding: BAAI/bge-small-en-v1.5
#   Semantic: 70%
#   BM25: 30%
#   Top-K: 5
#   Reranker: None
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(BASE_DIR / "scripts" / "02_retrieval"))

from config.retrieval_config import (
    EMBEDDING_MODEL_NAME,
    ALPHA,
    FINAL_TOP_K,
)


VECTOR_DB_DIR = BASE_DIR / "data" / "vector_store"
COLLECTION_NAME = "pulmo_guide"

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
    / "threshold_experiment_results.json"
)

RESULTS_MD_PATH = (
    RESULTS_DIR
    / "threshold_experiment_report.md"
)


# ============================================================
# THRESHOLDS TO TEST
# ============================================================

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


# ============================================================
# HELPERS
# ============================================================

def to_base_id(chunk_id: str) -> str:

    parts = chunk_id.split("_")

    return "_".join(parts[:2])


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
# LOAD EVALUATION SET
# ============================================================

def load_evaluation_set():

    print("Loading evaluation set...")

    with open(
        EVAL_SET_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not data:

        raise ValueError(
            "Evaluation set is empty."
        )

    print(
        f"OK: Loaded {len(data)} questions."
    )

    return data


# ============================================================
# LOAD RETRIEVAL COMPONENTS
# ============================================================

def load_components():

    print("=" * 70)

    print("Loading embedding model...")

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

    data = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    chunk_ids = data["ids"]

    chunk_texts = data["documents"]

    chunk_metadata = data["metadatas"]

    print(
        f"OK: Loaded {len(chunk_texts)} chunks."
    )

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
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    components
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
    # Semantic
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
        ],
    )

    semantic_ids = (
        semantic_results["ids"][0]
    )

    semantic_distances = (
        semantic_results["distances"][0]
    )

    semantic_score_map = {

        chunk_id:
            1 - distance

        for chunk_id, distance
        in zip(
            semantic_ids,
            semantic_distances
        )
    }

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    tokens = query.lower().split()

    bm25_scores = bm25.get_scores(
        tokens
    )

    bm25_score_map = {

        chunk_id:
            float(score)

        for chunk_id, score
        in zip(
            chunk_ids,
            bm25_scores
        )
    }

    # --------------------------------------------------------
    # Align
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

        ALPHA
        * semantic_normalized

        +

        (1 - ALPHA)
        * keyword_normalized

    )

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    indices = np.argsort(
        hybrid_scores
    )[::-1][:FINAL_TOP_K]

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

            "base_id":
                to_base_id(
                    chunk_ids[index]
                ),

            "score":
                float(
                    hybrid_scores[index]
                ),

            "semantic_score":
                float(
                    semantic_normalized[index]
                ),

            "bm25_score":
                float(
                    keyword_normalized[index]
                ),

            "text":
                chunk_texts[index],

            "metadata":
                chunk_metadata[index],
        })

    return results


# ============================================================
# EVALUATE THRESHOLDS
# ============================================================

def evaluate_thresholds(
    question_results
):

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD ANALYSIS"
    )

    print(
        "=" * 70
    )

    threshold_results = []

    total_questions = len(
        question_results
    )

    for threshold in THRESHOLDS:

        accepted = 0

        rejected = 0

        correct_accepts = 0

        false_accepts = 0

        missed_relevant = 0

        for record in question_results:

            top_score = record[
                "top_score"
            ]

            has_relevant = record[
                "has_relevant_chunk"
            ]

            passes = (
                top_score >= threshold
            )

            if passes:

                accepted += 1

                if has_relevant:

                    correct_accepts += 1

                else:

                    false_accepts += 1

            else:

                rejected += 1

                if has_relevant:

                    missed_relevant += 1

        acceptance_rate = (

            accepted
            / total_questions

            if total_questions
            else 0

        )

        precision_of_accepts = (

            correct_accepts
            / accepted

            if accepted
            else 0

        )

        recall_of_relevant = (

            correct_accepts
            / (
                correct_accepts
                + missed_relevant
            )

            if (
                correct_accepts
                + missed_relevant
            )
            else 0

        )

        threshold_results.append({

            "threshold":
                threshold,

            "accepted":
                accepted,

            "rejected":
                rejected,

            "acceptance_rate":
                acceptance_rate,

            "correct_accepts":
                correct_accepts,

            "false_accepts":
                false_accepts,

            "missed_relevant":
                missed_relevant,

            "accept_precision":
                precision_of_accepts,

            "relevant_recall":
                recall_of_relevant,
        })

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"{'Threshold':<12}"
        f"{'Accept':<10}"
        f"{'Reject':<10}"
        f"{'Accept %':<12}"
        f"{'Correct':<10}"
        f"{'False':<10}"
        f"{'Missed':<10}"
        f"{'Precision':<12}"
        f"{'Recall':<10}"
    )

    print(
        "-" * 100
    )

    for result in threshold_results:

        print(

            f"{result['threshold']:<12.2f}"

            f"{result['accepted']:<10}"

            f"{result['rejected']:<10}"

            f"{result['acceptance_rate']:<12.3f}"

            f"{result['correct_accepts']:<10}"

            f"{result['false_accepts']:<10}"

            f"{result['missed_relevant']:<10}"

            f"{result['accept_precision']:<12.3f}"

            f"{result['relevant_recall']:<10.3f}"
        )

    return threshold_results


# ============================================================
# MARKDOWN REPORT
# ============================================================

def build_markdown_report(
    question_results,
    threshold_results
):

    lines = []

    lines.append(
        "# Pulmo Guide — Threshold Experiment"
    )

    lines.append("")

    lines.append(
        "## Retrieval Configuration"
    )

    lines.append("")

    lines.append(
        f"- Embedding: `{EMBEDDING_MODEL_NAME}`"
    )

    lines.append(
        "- Semantic weight: `70%`"
    )

    lines.append(
        "- BM25 weight: `30%`"
    )

    lines.append(
        f"- Top-K: `{FINAL_TOP_K}`"
    )

    lines.append(
        "- Reranker: `None`"
    )

    lines.append("")

    lines.append(
        "## Threshold Comparison"
    )

    lines.append("")

    lines.append(
        "| Threshold | Accepted | Rejected | "
        "Acceptance % | Correct | False | "
        "Missed | Precision | Recall |"
    )

    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for result in threshold_results:

        lines.append(

            f"| {result['threshold']:.2f} | "
            f"{result['accepted']} | "
            f"{result['rejected']} | "
            f"{result['acceptance_rate']:.3f} | "
            f"{result['correct_accepts']} | "
            f"{result['false_accepts']} | "
            f"{result['missed_relevant']} | "
            f"{result['accept_precision']:.3f} | "
            f"{result['relevant_recall']:.3f} |"
        )

    lines.append("")

    lines.append(
        "## Per-Question Top Scores"
    )

    lines.append("")

    for i, record in enumerate(
        question_results,
        start=1
    ):

        lines.append(
            f"### {i}. {record['question']}"
        )

        lines.append("")

        lines.append(
            f"- Top-1 hybrid score: "
            f"`{record['top_score']:.4f}`"
        )

        lines.append(
            f"- Relevant chunk retrieved: "
            f"`{record['has_relevant_chunk']}`"
        )

        lines.append(
            f"- Ground truth: "
            f"`{record['ground_truth_ids']}`"
        )

        lines.append("")

        lines.append(
            "| Rank | Chunk ID | Score | Relevant |"
        )

        lines.append(
            "|---:|---|---:|---|"
        )

        for result in record["results"]:

            relevant = (
                result["base_id"]
                in set(record["ground_truth_ids"])
            )

            lines.append(

                f"| {result['rank']} | "
                f"`{result['base_id']}` | "
                f"{result['score']:.4f} | "
                f"{relevant} |"
            )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "PULMO GUIDE - THRESHOLD EXPERIMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"Embedding Model : "
        f"{EMBEDDING_MODEL_NAME}"
    )

    print(
        "Retrieval       : Hybrid"
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
        f"Evaluation Set  : "
        f"{EVAL_SET_PATH}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    evaluation_set = (
        load_evaluation_set()
    )

    components = (
        load_components()
    )

    # --------------------------------------------------------
    # Run retrieval
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING HYBRID RETRIEVAL"
    )

    print(
        "=" * 70
    )

    question_results = []

    for i, item in enumerate(
        evaluation_set,
        start=1
    ):

        question = item[
            "question"
        ]

        ground_truth = set(
            item[
                "relevant_chunk_ids"
            ]
        )

        results = hybrid_search(
            question,
            components
        )

        top_score = results[0][
            "score"
        ]

        has_relevant = any(

            result["base_id"]
            in ground_truth

            for result in results

        )

        record = {

            "question":
                question,

            "ground_truth_ids":
                item[
                    "relevant_chunk_ids"
                ],

            "top_score":
                top_score,

            "has_relevant_chunk":
                has_relevant,

            "results":
                results,
        }

        question_results.append(
            record
        )

        print(

            f"[{i}/{len(evaluation_set)}] "
            f"Top score: "
            f"{top_score:.4f} | "
            f"Relevant: "
            f"{has_relevant} | "
            f"{question[:60]}..."
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    threshold_results = (
        evaluate_thresholds(
            question_results
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    payload = {

        "configuration": {

            "embedding_model":
                EMBEDDING_MODEL_NAME,

            "semantic_weight":
                0.70,

            "bm25_weight":
                0.30,

            "top_k":
                FINAL_TOP_K,

            "reranker":
                None,
        },

        "thresholds_tested":
            THRESHOLDS,

        "threshold_results":
            threshold_results,

        "question_results":
            question_results,
    }

    with open(
        RESULTS_JSON_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False
        )

    report = build_markdown_report(
        question_results,
        threshold_results
    )

    with open(
        RESULTS_MD_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD EXPERIMENT COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Saved JSON: "
        f"{RESULTS_JSON_PATH}"
    )

    print(
        f"Saved Markdown: "
        f"{RESULTS_MD_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT change retrieval_config.py yet."
    )

    print(
        "Choose the threshold only after reviewing "
        "the experiment results."
    )


if __name__ == "__main__":
    main()
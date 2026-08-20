"""
============================================================
PULMO GUIDE
DAY 4 — FAITHFULNESS EVALUATION
============================================================

Purpose:
    Evaluate whether generated answers stay faithful to
    the available Core / Patient evidence.

Design:
    - No LangChain
    - No LLM required
    - Deterministic
    - Core evidence loaded from ChromaDB
    - Patient evidence loaded from data/processed/patient/
    - Semantic similarity
    - Token overlap
    - Numeric matching
    - Safe to run independently
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
EVALUATION_DIR = DATA_DIR / "evaluation"
PROCESSED_DIR = DATA_DIR / "processed"

CORE_DIR = PROCESSED_DIR / "core"
PATIENT_DIR = PROCESSED_DIR / "patient"

VECTOR_STORE_DIR = DATA_DIR / "vector_store"

RESULTS_FILE = (
    EVALUATION_DIR / "faithfulness_results.json"
)

REPORT_FILE = (
    EVALUATION_DIR / "faithfulness_report.txt"
)

CORE_COLLECTION_NAME = "pulmo_guide"


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

SEMANTIC_THRESHOLD = 0.55
TOKEN_OVERLAP_THRESHOLD = 0.20
FAITHFULNESS_THRESHOLD = 0.55


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "id": "FA-01",
        "source": "core",
        "question": "What are the symptoms of lung cancer?",
        "answer": (
            "Common symptoms of lung cancer can include a persistent "
            "cough, coughing up blood, chest pain, and breathlessness."
        ),
    },
    {
        "id": "FA-02",
        "source": "core",
        "question": (
            "What treatment options are recommended for people "
            "with lung cancer?"
        ),
        "answer": (
            "Treatment options depend on the type and stage of lung "
            "cancer and may include surgery, radiotherapy, systemic "
            "anticancer treatment, or combinations of these approaches."
        ),
    },
    {
        "id": "FA-03",
        "source": "core",
        "question": (
            "What imaging should be offered to people with stage 3 NSCLC?"
        ),
        "answer": (
            "Imaging recommendations for people with stage 3 NSCLC "
            "depend on the clinical situation and the extent of disease."
        ),
    },
    {
        "id": "FA-04",
        "source": "patient",
        "question": "What is my FEV1?",
        "answer": (
            "Your report records an FEV1 of 1.86 L, which is "
            "76% of the predicted value."
        ),
    },
    {
        "id": "FA-05",
        "source": "core+patient",
        "question": "What does this result mean?",
        "answer": (
            "The report describes a mild restrictive ventilatory "
            "pattern with mildly reduced diffusion capacity. "
            "FEV1 is 76% predicted and TLCO is also 76% predicted."
        ),
    },
    {
        "id": "FA-06",
        "source": "core+patient",
        "question": "Is this result normal?",
        "answer": (
            "The report does not describe the result as completely "
            "normal. It describes a mild restrictive ventilatory "
            "pattern and mildly reduced diffusion capacity."
        ),
    },
    {
        "id": "FA-07",
        "source": "core",
        "question": (
            "What is the recommended treatment for pancreatic cancer?"
        ),
        "answer": (
            "I cannot provide a recommendation for pancreatic cancer "
            "because the available guideline is focused on lung cancer."
        ),
    },
    {
        "id": "FA-08",
        "source": "core+patient",
        "question": "What does my result mean?",
        "answer": (
            "The available report provides findings that can be "
            "described from the supplied document, but interpretation "
            "should remain limited to the available evidence."
        ),
    },
]


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text for deterministic comparison."""

    if not text:
        return ""

    text = str(text).lower()

    # Keep medical terms, numbers, percentages and decimals.
    text = re.sub(
        r"[^a-z0-9%./\-]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def tokenize(text: str) -> set[str]:
    """Return normalized token set."""

    text = normalize_text(text)

    if not text:
        return set()

    return set(text.split())


def token_overlap(
    answer: str,
    evidence: str,
) -> float:
    """
    Directional token overlap.

    Measures how much of the answer vocabulary
    appears in the evidence.
    """

    answer_tokens = tokenize(answer)
    evidence_tokens = tokenize(evidence)

    if not answer_tokens:
        return 0.0

    return (
        len(answer_tokens & evidence_tokens)
        / len(answer_tokens)
    )


def extract_numbers(text: str) -> List[str]:
    """Extract numbers including percentages and decimals."""

    if not text:
        return []

    return re.findall(
        r"\b\d+(?:\.\d+)?\s*%?",
        str(text),
    )


def numeric_match(
    answer: str,
    evidence: str,
) -> bool:
    """
    Check whether numerical claims in the answer
    are present in the evidence.
    """

    answer_numbers = extract_numbers(answer)

    # No numbers = nothing to verify.
    if not answer_numbers:
        return True

    evidence_numbers = extract_numbers(evidence)

    normalized_evidence = {
        number.replace(" ", "")
        for number in evidence_numbers
    }

    for number in answer_numbers:

        if (
            number.replace(" ", "")
            not in normalized_evidence
        ):
            return False

    return True


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path: Path) -> Any:
    """Safely load JSON."""

    if not path.exists():
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return None


def extract_chunks(
    data: Any,
) -> List[Dict[str, Any]]:
    """
    Support the JSON structures used by the project.
    """

    if data is None:
        return []

    # Direct list of chunks.
    if isinstance(data, list):

        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    # Dictionary containing chunks.
    if isinstance(data, dict):

        for key in (
            "chunks",
            "documents",
            "data",
            "results",
            "patient_chunks",
            "core_chunks",
            "items",
        ):

            value = data.get(key)

            if isinstance(value, list):

                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

    return []


def get_chunk_text(
    chunk: Dict[str, Any],
) -> str:
    """Extract text from a chunk."""

    for key in (
        "text",
        "content",
        "chunk_text",
        "document",
        "page_content",
    ):

        value = chunk.get(key)

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value

    return ""


def get_chunk_id(
    chunk: Dict[str, Any],
) -> str:
    """Extract chunk ID."""

    for key in (
        "chunk_id",
        "id",
        "document_id",
    ):

        value = chunk.get(key)

        if value is not None:
            return str(value)

    return "unknown"


# ============================================================
# CORE — CHROMADB
# ============================================================

def load_core_from_chroma() -> List[Dict[str, Any]]:
    """
    Load Core evidence from the existing ChromaDB.

    This uses the same Core collection used by the project.
    """

    try:

        import chromadb

    except ImportError:

        print(
            "ERROR: chromadb is not installed."
        )

        return []

    if not VECTOR_STORE_DIR.exists():

        print(
            "WARNING: Core vector store not found:"
        )

        print(
            f"  {VECTOR_STORE_DIR}"
        )

        return []

    try:

        client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_DIR)
        )

        collection = client.get_collection(
            name=CORE_COLLECTION_NAME
        )

        total = collection.count()

        if total == 0:
            return []

        data = collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        documents = (
            data.get("documents")
            or []
        )

        metadatas = (
            data.get("metadatas")
            or []
        )

        ids = (
            data.get("ids")
            or []
        )

        chunks = []

        for index, document in enumerate(
            documents
        ):

            if not document:
                continue

            metadata = (
                metadatas[index]
                if (
                    index < len(metadatas)
                    and isinstance(
                        metadatas[index],
                        dict,
                    )
                )
                else {}
            )

            chunk_id = (
                ids[index]
                if index < len(ids)
                else metadata.get(
                    "chunk_id",
                    f"core_{index:04d}",
                )
            )

            chunk = {
                "chunk_id": str(chunk_id),
                "text": str(document),
                "source": "core",
                **metadata,
            }

            chunks.append(chunk)

        return chunks

    except Exception as error:

        print(
            "WARNING: Could not load Core ChromaDB."
        )

        print(
            f"  {error}"
        )

        return []


# ============================================================
# PATIENT JSON DISCOVERY
# ============================================================

def find_json_files(
    directory: Path,
) -> List[Path]:
    """
    Find all JSON files recursively inside a directory.
    """

    if not directory.exists():
        return []

    return sorted(
        directory.rglob("*.json")
    )


def load_evidence_directory(
    directory: Path,
    source_name: str,
) -> List[Dict[str, Any]]:
    """
    Load evidence JSON files recursively.

    This is intentionally flexible because the project
    stores Patient evidence inside data/processed/patient/
    rather than a single patient_chunks.json file.
    """

    chunks: List[Dict[str, Any]] = []

    json_files = find_json_files(
        directory
    )

    for json_file in json_files:

        data = load_json(
            json_file
        )

        if data is None:
            continue

        file_chunks = extract_chunks(
            data
        )

        for chunk in file_chunks:

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            text = get_chunk_text(
                chunk
            )

            if not text.strip():
                continue

            normalized_chunk = dict(
                chunk
            )

            normalized_chunk.setdefault(
                "source",
                source_name,
            )

            normalized_chunk.setdefault(
                "chunk_id",
                get_chunk_id(
                    normalized_chunk
                ),
            )

            chunks.append(
                normalized_chunk
            )

    return chunks


# ============================================================
# PATIENT EVIDENCE
# ============================================================

def load_patient_evidence() -> List[Dict[str, Any]]:
    """
    Load Patient evidence from:

        data/processed/patient/

    recursively.
    """

    return load_evidence_directory(
        PATIENT_DIR,
        "patient",
    )


# ============================================================
# EVIDENCE LOADING
# ============================================================

def load_evidence() -> Dict[
    str,
    List[Dict[str, Any]]
]:
    """
    Load the actual project evidence.

    Core:
        ChromaDB

    Patient:
        data/processed/patient/**/*.json
    """

    core_chunks = (
        load_core_from_chroma()
    )

    patient_chunks = (
        load_patient_evidence()
    )

    return {
        "core": core_chunks,
        "patient": patient_chunks,
    }


# ============================================================
# EMBEDDING MODEL
# ============================================================

def load_embedding_model(
) -> SentenceTransformer:
    """Load the project's embedding model."""

    print(
        "Loading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(
        "OK: Embedding model loaded."
    )

    return model


# ============================================================
# EVIDENCE POOL
# ============================================================

def select_evidence_pool(
    source: str,
    evidence: Dict[
        str,
        List[Dict[str, Any]]
    ],
) -> List[Dict[str, Any]]:
    """Select evidence according to source mode."""

    if source == "core":
        return evidence["core"]

    if source == "patient":
        return evidence["patient"]

    if source == "core+patient":

        return (
            evidence["core"]
            + evidence["patient"]
        )

    return (
        evidence["core"]
        + evidence["patient"]
    )


# ============================================================
# BEST EVIDENCE
# ============================================================

def retrieve_best_evidence(
    answer: str,
    source: str,
    evidence: Dict[
        str,
        List[Dict[str, Any]]
    ],
    model: SentenceTransformer,
) -> Dict[str, Any]:
    """
    Find the evidence chunk most semantically related
    to the answer.
    """

    pool = select_evidence_pool(
        source,
        evidence,
    )

    valid_chunks = [
        chunk
        for chunk in pool
        if get_chunk_text(
            chunk
        ).strip()
    ]

    if not valid_chunks:

        return {
            "chunk": None,
            "semantic_similarity": 0.0,
            "token_overlap": 0.0,
            "numeric_match": False,
            "score": 0.0,
        }

    answer_embedding = model.encode(
        answer,
        normalize_embeddings=True,
    )

    evidence_texts = [
        get_chunk_text(chunk)
        for chunk in valid_chunks
    ]

    evidence_embeddings = model.encode(
        evidence_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarities = np.dot(
        evidence_embeddings,
        answer_embedding,
    )

    best_index = int(
        np.argmax(similarities)
    )

    best_chunk = (
        valid_chunks[best_index]
    )

    best_text = (
        evidence_texts[best_index]
    )

    semantic_score = float(
        similarities[best_index]
    )

    overlap_score = token_overlap(
        answer,
        best_text,
    )

    numbers_ok = numeric_match(
        answer,
        best_text,
    )

    # Deterministic combined score.
    score = (
        0.60 * semantic_score
        + 0.30 * overlap_score
        + 0.10 * float(numbers_ok)
    )

    return {
        "chunk": best_chunk,
        "semantic_similarity": round(
            semantic_score,
            4,
        ),
        "token_overlap": round(
            overlap_score,
            4,
        ),
        "numeric_match": numbers_ok,
        "score": round(
            float(score),
            4,
        ),
    }


# ============================================================
# FAITHFULNESS DECISION
# ============================================================

def evaluate_faithfulness(
    answer: str,
    source: str,
    evidence: Dict[
        str,
        List[Dict[str, Any]]
    ],
    model: SentenceTransformer,
) -> Dict[str, Any]:
    """
    Evaluate answer faithfulness.
    """

    answer_lower = answer.lower()

    # --------------------------------------------------------
    # Scope-safe refusal
    # --------------------------------------------------------

    refusal_markers = [
        "cannot provide",
        "out of scope",
        "focused on lung cancer",
        "does not cover",
        "not covered",
    ]

    if any(
        marker in answer_lower
        for marker in refusal_markers
    ):

        return {
            "result": "FAITHFUL",
            "score": 1.0,
            "semantic_similarity": 1.0,
            "token_overlap": 1.0,
            "numeric_match": True,
            "evidence": None,
            "reason": (
                "Scope-safe refusal."
            ),
        }

    # --------------------------------------------------------
    # Retrieve evidence
    # --------------------------------------------------------

    retrieval = retrieve_best_evidence(
        answer=answer,
        source=source,
        evidence=evidence,
        model=model,
    )

    score = retrieval[
        "score"
    ]

    semantic = retrieval[
        "semantic_similarity"
    ]

    overlap = retrieval[
        "token_overlap"
    ]

    numbers_ok = retrieval[
        "numeric_match"
    ]

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if (
        score >= FAITHFULNESS_THRESHOLD
        and semantic >= SEMANTIC_THRESHOLD
        and overlap >= TOKEN_OVERLAP_THRESHOLD
        and numbers_ok
    ):

        result = "FAITHFUL"

    elif (
        score >= 0.45
        or semantic >= 0.50
    ):

        result = "PARTIALLY_FAITHFUL"

    else:

        result = "UNFAITHFUL"

    # --------------------------------------------------------
    # Evidence metadata
    # --------------------------------------------------------

    chunk = retrieval[
        "chunk"
    ]

    evidence_info = None

    if chunk is not None:

        evidence_info = {
            "chunk_id": get_chunk_id(
                chunk
            ),
            "source": chunk.get(
                "source",
                source,
            ),
            "page": chunk.get(
                "page",
                chunk.get(
                    "page_number"
                ),
            ),
            "section": chunk.get(
                "section",
                chunk.get(
                    "section_title"
                ),
            ),
            "text": get_chunk_text(
                chunk
            ),
        }

    # --------------------------------------------------------
    # Reason
    # --------------------------------------------------------

    if result == "FAITHFUL":

        reason = (
            "Answer is sufficiently grounded "
            "in the available evidence."
        )

    elif result == "PARTIALLY_FAITHFUL":

        reason = (
            "Answer has partial grounding "
            "in the available evidence."
        )

    else:

        reason = (
            "Answer is not sufficiently grounded "
            "in the available evidence."
        )

    return {
        "result": result,
        "score": score,
        "semantic_similarity": semantic,
        "token_overlap": overlap,
        "numeric_match": numbers_ok,
        "evidence": evidence_info,
        "reason": reason,
    }


# ============================================================
# REPORT
# ============================================================

def write_report(
    results: List[Dict[str, Any]]
) -> None:

    total = len(results)

    faithful = sum(
        result["result"]
        == "FAITHFUL"
        for result in results
    )

    partial = sum(
        result["result"]
        == "PARTIALLY_FAITHFUL"
        for result in results
    )

    unfaithful = sum(
        result["result"]
        == "UNFAITHFUL"
        for result in results
    )

    accuracy = (
        faithful / total * 100
        if total
        else 0.0
    )

    lines = [
        "=" * 70,
        "PULMO GUIDE — DAY 4",
        "FAITHFULNESS EVALUATION",
        "=" * 70,
        "",
        f"Total tests: {total}",
        f"Faithful: {faithful}",
        f"Partially faithful: {partial}",
        f"Unfaithful: {unfaithful}",
        (
            "Faithfulness accuracy: "
            f"{accuracy:.2f}%"
        ),
        "",
    ]

    for result in results:

        lines.append(
            "-" * 70
        )

        lines.append(
            f'{result["id"]} — '
            f'{result["source"]}'
        )

        lines.append(
            f'Question: '
            f'{result["question"]}'
        )

        lines.append(
            f'Answer: '
            f'{result["answer"]}'
        )

        lines.append(
            f'Result: '
            f'{result["result"]}'
        )

        lines.append(
            f'Score: '
            f'{result["score"]}'
        )

        lines.append(
            "Semantic similarity: "
            f'{result["semantic_similarity"]}'
        )

        lines.append(
            "Token overlap: "
            f'{result["token_overlap"]}'
        )

        lines.append(
            "Numeric match: "
            f'{result["numeric_match"]}'
        )

        lines.append(
            f'Reason: '
            f'{result["reason"]}'
        )

        evidence = result.get(
            "evidence"
        )

        if evidence:

            lines.append(
                "Evidence: "
                f'{evidence["chunk_id"]}'
            )

            lines.append(
                "Source: "
                f'{evidence["source"]}'
            )

            lines.append(
                "Page: "
                f'{evidence["page"]}'
            )

            lines.append(
                "Section: "
                f'{evidence["section"]}'
            )

        lines.append("")

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PULMO GUIDE — DAY 4")
    print("FAITHFULNESS EVALUATION")
    print("=" * 70)
    print()

    print(
        "Initializing faithfulness evaluation..."
    )

    print(
        "INFO: No LangChain required."
    )

    print(
        "Using deterministic faithfulness evaluation."
    )

    print()

    # --------------------------------------------------------
    # Load evidence
    # --------------------------------------------------------

    print(
        "Loading evidence..."
    )

    evidence = load_evidence()

    core_count = len(
        evidence["core"]
    )

    patient_count = len(
        evidence["patient"]
    )

    print(
        "Core evidence chunks: "
        f"{core_count}"
    )

    print(
        "Patient evidence chunks: "
        f"{patient_count}"
    )

    print(
        "Core + Patient: "
        f"{core_count + patient_count}"
    )

    if (
        core_count == 0
        and patient_count == 0
    ):

        print()

        print(
            "ERROR: No evidence could be loaded."
        )

        print(
            "Expected:"
        )

        print(
            f"  Core: {CORE_DIR}"
        )

        print(
            f"  Patient: {PATIENT_DIR}"
        )

        print(
            f"  ChromaDB: {VECTOR_STORE_DIR}"
        )

        sys.exit(1)

    print()

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    model = load_embedding_model()

    print()

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    print(
        "Total faithfulness tests: "
        f"{len(TEST_CASES)}"
    )

    results = []

    for test in TEST_CASES:

        print(
            "-" * 70
        )

        print(
            f'{test["id"]} — '
            f'{test["source"]}'
        )

        print(
            f'Question: '
            f'{test["question"]}'
        )

        evaluation = (
            evaluate_faithfulness(
                answer=test["answer"],
                source=test["source"],
                evidence=evidence,
                model=model,
            )
        )

        result = {
            "id": test["id"],
            "source": test["source"],
            "question": test["question"],
            "answer": test["answer"],
            **evaluation,
        }

        results.append(result)

        print(
            f'Result: '
            f'{evaluation["result"]}'
        )

        print(
            f'Score: '
            f'{evaluation["score"]}'
        )

        print(
            "Semantic similarity: "
            f'{evaluation["semantic_similarity"]}'
        )

        print(
            "Token overlap: "
            f'{evaluation["token_overlap"]}'
        )

        print(
            "Numeric match: "
            f'{evaluation["numeric_match"]}'
        )

        if evaluation.get(
            "evidence"
        ):

            print(
                "Evidence: "
                f'{evaluation["evidence"]["chunk_id"]}'
            )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    total = len(results)

    faithful = sum(
        result["result"]
        == "FAITHFUL"
        for result in results
    )

    partial = sum(
        result["result"]
        == "PARTIALLY_FAITHFUL"
        for result in results
    )

    unfaithful = sum(
        result["result"]
        == "UNFAITHFUL"
        for result in results
    )

    accuracy = (
        faithful / total * 100
        if total
        else 0.0
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output = {
        "project": "Pulmo Guide",
        "day": 4,
        "evaluation": "faithfulness",
        "method": "deterministic",
        "langchain_used": False,
        "llm_required": False,
        "embedding_model": (
            EMBEDDING_MODEL_NAME
        ),
        "evidence_sources": {
            "core": "ChromaDB",
            "patient": (
                "data/processed/patient/"
            ),
        },
        "thresholds": {
            "semantic_similarity": (
                SEMANTIC_THRESHOLD
            ),
            "token_overlap": (
                TOKEN_OVERLAP_THRESHOLD
            ),
            "faithfulness_score": (
                FAITHFULNESS_THRESHOLD
            ),
        },
        "evidence_counts": {
            "core": core_count,
            "patient": patient_count,
            "combined": (
                core_count
                + patient_count
            ),
        },
        "summary": {
            "total_tests": total,
            "faithful": faithful,
            "partially_faithful": partial,
            "unfaithful": unfaithful,
            "faithfulness_accuracy": round(
                accuracy,
                2,
            ),
        },
        "results": results,
    }

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    write_report(
        results
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Total tests: {total}"
    )

    print(
        f"Faithful: {faithful}"
    )

    print(
        f"Partially faithful: {partial}"
    )

    print(
        f"Unfaithful: {unfaithful}"
    )

    print(
        "Faithfulness accuracy: "
        f"{accuracy:.2f}%"
    )

    print()

    print(
        "JSON saved to:"
    )

    print(
        RESULTS_FILE
    )

    print()

    print(
        "Report saved to:"
    )

    print(
        REPORT_FILE
    )

    print()

    print(
        "Faithfulness evaluation completed."
    )


if __name__ == "__main__":
    main()
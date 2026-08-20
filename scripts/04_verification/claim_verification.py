"""
============================================================
PULMO GUIDE
DAY 4 — CLAIM VERIFICATION
============================================================

Purpose:
    Verify whether extracted claims are supported by the
    available Core / Patient evidence.

Design:
    - No LangChain
    - No LLM required
    - Deterministic verification
    - Uses BGE semantic similarity
    - Uses token overlap
    - Uses numeric consistency
    - Supports Core / Patient / Core+Patient evidence
    - Reads evidence recursively from:
        data/processed/core/
        data/processed/patient/
============================================================
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CORE_DIR = PROCESSED_DIR / "core"
PATIENT_DIR = PROCESSED_DIR / "patient"

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

CLAIM_RESULTS_FILE = (
    EVALUATION_DIR / "claim_extraction_results.json"
)

RESULTS_FILE = (
    EVALUATION_DIR / "claim_verification_results.json"
)

REPORT_FILE = (
    EVALUATION_DIR / "claim_verification_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Verification thresholds
SEMANTIC_THRESHOLD = 0.55
TOKEN_OVERLAP_THRESHOLD = 0.20
SUPPORT_SCORE_THRESHOLD = 0.55

# Strong support threshold
STRONG_SEMANTIC_THRESHOLD = 0.70
STRONG_TOKEN_THRESHOLD = 0.30


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "id": "CE-01",
        "source": "core",
        "question": "What are the symptoms of lung cancer?",
        "claims": [
            (
                "Common symptoms of lung cancer can include a persistent "
                "cough, coughing up blood, chest pain, and breathlessness."
            ),
            (
                "The guideline should be consulted for the complete list "
                "and clinical context."
            ),
        ],
    },
    {
        "id": "CE-02",
        "source": "core",
        "question": (
            "What treatment options are recommended for people "
            "with lung cancer?"
        ),
        "claims": [
            (
                "Treatment options depend on the type and stage of lung "
                "cancer and may include surgery, radiotherapy, systemic "
                "anticancer treatment, or combinations of these approaches."
            ),
            (
                "The appropriate option depends on the patient's "
                "clinical assessment."
            ),
        ],
    },
    {
        "id": "CE-03",
        "source": "core",
        "question": (
            "What imaging should be offered to people with stage 3 NSCLC?"
        ),
        "claims": [
            (
                "Imaging recommendations for people with stage 3 NSCLC "
                "depend on the clinical situation and the extent of disease."
            ),
            (
                "The relevant guideline recommendations should be followed "
                "for selecting appropriate imaging."
            ),
        ],
    },
    {
        "id": "CE-04",
        "source": "patient",
        "question": "What is my FEV1?",
        "claims": [
            (
                "Your report records an FEV1 of 1.86 L, which is "
                "76% of the predicted value."
            ),
        ],
    },
    {
        "id": "CE-05",
        "source": "core+patient",
        "question": "What does this result mean?",
        "claims": [
            (
                "The report describes a mild restrictive ventilatory "
                "pattern with mildly reduced diffusion capacity."
            ),
            (
                "The measured FEV1 is 76% of predicted and TLCO is also "
                "76% of predicted."
            ),
            (
                "The report states that these findings do not preclude "
                "consideration of treatment with curative intent."
            ),
        ],
    },
    {
        "id": "CE-06",
        "source": "core+patient",
        "question": "Is this result normal?",
        "claims": [
            (
                "The report does not describe the result as completely normal."
            ),
            (
                "It describes a mild restrictive ventilatory pattern and "
                "mildly reduced diffusion capacity."
            ),
        ],
    },
    {
        "id": "CE-07",
        "source": "core",
        "question": (
            "What is the recommended treatment for pancreatic cancer?"
        ),
        "claims": [],
    },
    {
        "id": "CE-08",
        "source": "core+patient",
        "question": "What does my result mean?",
        "claims": [
            (
                "The available report provides findings that can be "
                "described from the document, but interpretation should "
                "remain limited to the supplied evidence."
            ),
        ],
    },
]


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower()

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
    normalized = normalize_text(text)

    if not normalized:
        return set()

    return set(normalized.split())


def token_overlap(
    claim: str,
    evidence: str,
) -> float:

    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence)

    if not claim_tokens:
        return 0.0

    return len(
        claim_tokens & evidence_tokens
    ) / len(claim_tokens)


# ============================================================
# NUMERIC CHECKING
# ============================================================

def extract_numbers(text: str) -> List[str]:

    if not text:
        return []

    return re.findall(
        r"\b\d+(?:\.\d+)?\s*%?",
        str(text),
    )


def normalize_number(number: str) -> str:

    return (
        number
        .replace(" ", "")
        .strip()
    )


def numeric_match(
    claim: str,
    evidence: str,
) -> bool:

    claim_numbers = extract_numbers(claim)

    # No numbers in claim = no numeric conflict.
    if not claim_numbers:
        return True

    evidence_numbers = extract_numbers(
        evidence
    )

    evidence_set = {
        normalize_number(n)
        for n in evidence_numbers
    }

    for number in claim_numbers:

        if normalize_number(number) not in evidence_set:
            return False

    return True


# ============================================================
# JSON LOADING
# ============================================================

def load_json(path: Path) -> Any:

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return None


# ============================================================
# CHUNK EXTRACTION
# ============================================================

def extract_chunks(
    data: Any,
) -> List[Dict[str, Any]]:

    if data is None:
        return []

    # Direct list of chunks
    if isinstance(data, list):

        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    # Dictionary containing chunks
    if isinstance(data, dict):

        possible_keys = [
            "chunks",
            "documents",
            "data",
            "results",
            "core_chunks",
            "patient_chunks",
        ]

        for key in possible_keys:

            value = data.get(key)

            if isinstance(value, list):

                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

        # Sometimes a single chunk is stored as a dict.
        if any(
            key in data
            for key in (
                "text",
                "content",
                "chunk_text",
                "page_content",
            )
        ):
            return [data]

    return []


# ============================================================
# TEXT EXTRACTION
# ============================================================

def get_chunk_text(
    chunk: Dict[str, Any],
) -> str:

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

    for key in (
        "chunk_id",
        "id",
        "document_id",
        "chunkId",
    ):

        value = chunk.get(key)

        if value is not None:

            return str(value)

    return "unknown"


def get_source(
    chunk: Dict[str, Any],
    default_source: str,
) -> str:

    value = chunk.get("source")

    if value:
        return str(value)

    return default_source


def get_page(
    chunk: Dict[str, Any],
) -> Any:

    for key in (
        "page",
        "page_number",
        "page_num",
    ):

        if key in chunk:
            return chunk.get(key)

    return None


def get_section(
    chunk: Dict[str, Any],
) -> Any:

    for key in (
        "section",
        "section_title",
        "heading",
    ):

        if key in chunk:
            return chunk.get(key)

    return None


# ============================================================
# RECURSIVE JSON DISCOVERY
# ============================================================

def discover_json_files(
    directory: Path,
) -> List[Path]:

    if not directory.exists():
        return []

    return sorted(
        directory.rglob("*.json")
    )


def load_chunks_from_directory(
    directory: Path,
    source: str,
) -> List[Dict[str, Any]]:

    all_chunks: List[Dict[str, Any]] = []

    json_files = discover_json_files(
        directory
    )

    for json_file in json_files:

        data = load_json(
            json_file
        )

        chunks = extract_chunks(
            data
        )

        for chunk in chunks:

            text = get_chunk_text(
                chunk
            )

            if not text.strip():
                continue

            # Preserve original metadata.
            chunk_copy = dict(chunk)

            # Guarantee source metadata.
            chunk_copy.setdefault(
                "source",
                source,
            )

            # Keep filename for traceability.
            chunk_copy.setdefault(
                "file",
                str(
                    json_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
            )

            all_chunks.append(
                chunk_copy
            )

    return all_chunks


# ============================================================
# EVIDENCE LOADING
# ============================================================

def load_evidence() -> Dict[
    str,
    List[Dict[str, Any]]
]:

    core_chunks = load_chunks_from_directory(
        CORE_DIR,
        "core",
    )

    patient_chunks = load_chunks_from_directory(
        PATIENT_DIR,
        "patient",
    )

    return {
        "core": core_chunks,
        "patient": patient_chunks,
    }


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
# EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print(
        f"Loading embedding model: "
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
# CLAIM VERIFICATION
# ============================================================

def verify_claim(
    claim: str,
    source: str,
    evidence: Dict[
        str,
        List[Dict[str, Any]]
    ],
    model: SentenceTransformer,
) -> Dict[str, Any]:

    pool = select_evidence_pool(
        source,
        evidence,
    )

    valid_chunks = [
        chunk
        for chunk in pool
        if get_chunk_text(chunk).strip()
    ]

    if not valid_chunks:

        return {
            "status": "UNSUPPORTED",
            "score": 0.0,
            "semantic_similarity": 0.0,
            "token_overlap": 0.0,
            "numeric_match": False,
            "evidence": None,
            "reason": "No evidence available.",
        }

    claim_embedding = model.encode(
        claim,
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
        claim_embedding,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Don't blindly use semantic similarity alone.
    # Calculate combined support score for every chunk.
    # --------------------------------------------------------

    candidates = []

    for index, chunk in enumerate(
        valid_chunks
    ):

        evidence_text = evidence_texts[index]

        semantic = float(
            similarities[index]
        )

        overlap = token_overlap(
            claim,
            evidence_text,
        )

        numbers_ok = numeric_match(
            claim,
            evidence_text,
        )

        # Numeric contradiction is heavily penalized.
        if not numbers_ok:

            numeric_component = 0.0

        else:

            numeric_component = 1.0

        score = (
            0.60 * semantic
            + 0.30 * overlap
            + 0.10 * numeric_component
        )

        candidates.append(
            {
                "chunk": chunk,
                "semantic_similarity": semantic,
                "token_overlap": overlap,
                "numeric_match": numbers_ok,
                "score": score,
            }
        )

    # Highest combined support score.
    best = max(
        candidates,
        key=lambda item: item["score"],
    )

    semantic = best[
        "semantic_similarity"
    ]

    overlap = best[
        "token_overlap"
    ]

    numbers_ok = best[
        "numeric_match"
    ]

    score = best["score"]

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    # Strong support
    if (
        score >= SUPPORT_SCORE_THRESHOLD
        and semantic >= SEMANTIC_THRESHOLD
        and overlap >= TOKEN_OVERLAP_THRESHOLD
        and numbers_ok
    ):

        status = "SUPPORTED"

        reason = (
            "Claim is supported by evidence "
            "using semantic similarity, token "
            "overlap and numeric consistency."
        )

    # Partial support
    elif (
        score >= 0.45
        or semantic >= 0.50
    ):

        status = "PARTIALLY_SUPPORTED"

        reason = (
            "Evidence has partial semantic or "
            "lexical support but does not fully "
            "satisfy the support threshold."
        )

    # No sufficient support
    else:

        status = "UNSUPPORTED"

        reason = (
            "No sufficiently relevant evidence "
            "was found for the claim."
        )

    chunk = best["chunk"]

    evidence_info = {
        "chunk_id": get_chunk_id(chunk),
        "source": get_source(
            chunk,
            source,
        ),
        "page": get_page(chunk),
        "section": get_section(chunk),
        "text": get_chunk_text(chunk),
        "file": chunk.get("file"),
    }

    return {
        "status": status,
        "score": round(
            float(score),
            4,
        ),
        "semantic_similarity": round(
            semantic,
            4,
        ),
        "token_overlap": round(
            overlap,
            4,
        ),
        "numeric_match": numbers_ok,
        "evidence": evidence_info,
        "reason": reason,
    }


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(
    test: Dict[str, Any],
    evidence: Dict[
        str,
        List[Dict[str, Any]]
    ],
    model: SentenceTransformer,
) -> Dict[str, Any]:

    claims = test.get(
        "claims",
        [],
    )

    # Out-of-scope/refusal
    if not claims:

        return {
            "id": test["id"],
            "source": test["source"],
            "question": test["question"],
            "claims": [],
            "overall": "NO_CLAIMS",
            "claim_count": 0,
            "supported_claims": 0,
            "partially_supported_claims": 0,
            "unsupported_claims": 0,
        }

    claim_results = []

    for index, claim in enumerate(
        claims,
        start=1,
    ):

        verification = verify_claim(
            claim=claim,
            source=test["source"],
            evidence=evidence,
            model=model,
        )

        claim_results.append(
            {
                "claim_id": (
                    f'{test["id"]}-CLM-{index:02d}'
                ),
                "claim": claim,
                **verification,
            }
        )

    supported = sum(
        r["status"] == "SUPPORTED"
        for r in claim_results
    )

    partial = sum(
        r["status"] == "PARTIALLY_SUPPORTED"
        for r in claim_results
    )

    unsupported = sum(
        r["status"] == "UNSUPPORTED"
        for r in claim_results
    )

    total_claims = len(
        claim_results
    )

    # Overall result
    if supported == total_claims:

        overall = "SUPPORTED"

    elif (
        supported > 0
        or partial > 0
    ):

        overall = "PARTIALLY_SUPPORTED"

    else:

        overall = "UNSUPPORTED"

    return {
        "id": test["id"],
        "source": test["source"],
        "question": test["question"],
        "claims": claim_results,
        "overall": overall,
        "claim_count": total_claims,
        "supported_claims": supported,
        "partially_supported_claims": partial,
        "unsupported_claims": unsupported,
    }


# ============================================================
# REPORT
# ============================================================

def write_report(
    results: List[Dict[str, Any]],
) -> None:

    total_tests = len(results)

    passed_tests = sum(
        r["overall"] in (
            "SUPPORTED",
            "NO_CLAIMS",
        )
        for r in results
    )

    total_claims = sum(
        r.get(
            "claim_count",
            0,
        )
        for r in results
    )

    supported_claims = sum(
        r.get(
            "supported_claims",
            0,
        )
        for r in results
    )

    partial_claims = sum(
        r.get(
            "partially_supported_claims",
            0,
        )
        for r in results
    )

    unsupported_claims = sum(
        r.get(
            "unsupported_claims",
            0,
        )
        for r in results
    )

    test_accuracy = (
        passed_tests / total_tests * 100
        if total_tests
        else 0
    )

    support_rate = (
        supported_claims / total_claims * 100
        if total_claims
        else 0
    )

    lines = []

    lines.append("=" * 70)
    lines.append(
        "PULMO GUIDE — DAY 4"
    )
    lines.append(
        "CLAIM VERIFICATION"
    )
    lines.append("=" * 70)
    lines.append("")

    lines.append(
        f"Total tests: {total_tests}"
    )

    lines.append(
        f"Passed: {passed_tests}"
    )

    lines.append(
        f"Failed: "
        f"{total_tests - passed_tests}"
    )

    lines.append(
        f"Test accuracy: "
        f"{test_accuracy:.2f}%"
    )

    lines.append("")

    lines.append(
        f"Total claims: {total_claims}"
    )

    lines.append(
        f"Supported claims: "
        f"{supported_claims}"
    )

    lines.append(
        f"Partially supported claims: "
        f"{partial_claims}"
    )

    lines.append(
        f"Unsupported claims: "
        f"{unsupported_claims}"
    )

    lines.append(
        f"Claim support rate: "
        f"{support_rate:.2f}%"
    )

    lines.append("")

    for result in results:

        lines.append("-" * 70)

        lines.append(
            f'{result["id"]} — '
            f'{result["source"]}'
        )

        lines.append(
            f'Question: '
            f'{result["question"]}'
        )

        lines.append(
            f'Claims: '
            f'{result["claim_count"]}'
        )

        lines.append(
            f'Overall: '
            f'{result["overall"]}'
        )

        for claim in result.get(
            "claims",
            [],
        ):

            lines.append(
                f'  {claim["claim_id"]}: '
                f'{claim["status"]}'
            )

            lines.append(
                f'    Score: '
                f'{claim["score"]}'
            )

            lines.append(
                f'    Semantic similarity: '
                f'{claim["semantic_similarity"]}'
            )

            lines.append(
                f'    Token overlap: '
                f'{claim["token_overlap"]}'
            )

            lines.append(
                f'    Numeric match: '
                f'{claim["numeric_match"]}'
            )

            if claim.get("evidence"):

                lines.append(
                    f'    Evidence: '
                    f'{claim["evidence"]["chunk_id"]}'
                )

            lines.append(
                f'    Reason: '
                f'{claim["reason"]}'
            )

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
    print(
        "PULMO GUIDE — DAY 4"
    )
    print(
        "CLAIM VERIFICATION"
    )
    print("=" * 70)
    print()

    print(
        "Initializing claim verification..."
    )

    print(
        "Using deterministic claim verification."
    )

    print()

    # --------------------------------------------------------
    # Load evidence
    # --------------------------------------------------------

    print(
        "Loading Core evidence..."
    )

    core_chunks = load_chunks_from_directory(
        CORE_DIR,
        "core",
    )

    print(
        f"OK: Core evidence chunks: "
        f"{len(core_chunks)}"
    )

    print(
        "Loading Patient evidence..."
    )

    patient_chunks = load_chunks_from_directory(
        PATIENT_DIR,
        "patient",
    )

    print(
        f"OK: Patient evidence chunks: "
        f"{len(patient_chunks)}"
    )

    evidence = {
        "core": core_chunks,
        "patient": patient_chunks,
    }

    print()

    print(
        "Evidence availability:"
    )

    print(
        f"  Core: "
        f"{len(core_chunks)}"
    )

    print(
        f"  Patient: "
        f"{len(patient_chunks)}"
    )

    print(
        f"  Core + Patient: "
        f"{len(core_chunks) + len(patient_chunks)}"
    )

    if not core_chunks and not patient_chunks:

        print()
        print(
            "ERROR: No evidence chunks found."
        )

        print(
            "Expected Core under:"
        )

        print(
            f"  {CORE_DIR}"
        )

        print(
            "Expected Patient under:"
        )

        print(
            f"  {PATIENT_DIR}"
        )

        sys.exit(1)

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_embedding_model()

    print()

    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    print(
        f"Total verification tests: "
        f"{len(TEST_CASES)}"
    )

    results = []

    for test in TEST_CASES:

        print("-" * 70)

        print(
            f'{test["id"]} — '
            f'{test["source"]}'
        )

        print(
            f'Question: '
            f'{test["question"]}'
        )

        print(
            f'Claims: '
            f'{len(test["claims"])}'
        )

        result = evaluate_test(
            test,
            evidence,
            model,
        )

        results.append(result)

        print(
            f'Overall: '
            f'{result["overall"]}'
        )

        for claim in result.get(
            "claims",
            [],
        ):

            print(
                f'  {claim["claim_id"]}: '
                f'{claim["status"]}'
            )

            if claim.get("evidence"):

                print(
                    f'    Evidence: '
                    f'{claim["evidence"]["chunk_id"]}'
                )

                print(
                    f'    Score: '
                    f'{claim["score"]}'
                )

                print(
                    f'    Token overlap: '
                    f'{claim["token_overlap"]}'
                )

                print(
                    f'    Semantic similarity: '
                    f'{claim["semantic_similarity"]}'
                )

                print(
                    f'    Numeric match: '
                    f'{claim["numeric_match"]}'
                )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_tests = len(results)

    passed_tests = sum(
        r["overall"] in (
            "SUPPORTED",
            "NO_CLAIMS",
        )
        for r in results
    )

    failed_tests = (
        total_tests
        - passed_tests
    )

    total_claims = sum(
        r["claim_count"]
        for r in results
    )

    supported_claims = sum(
        r["supported_claims"]
        for r in results
    )

    partial_claims = sum(
        r["partially_supported_claims"]
        for r in results
    )

    unsupported_claims = sum(
        r["unsupported_claims"]
        for r in results
    )

    test_accuracy = (
        passed_tests
        / total_tests
        * 100
        if total_tests
        else 0.0
    )

    claim_support_rate = (
        supported_claims
        / total_claims
        * 100
        if total_claims
        else 0.0
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output = {
        "project": "Pulmo Guide",
        "day": 4,
        "evaluation": "claim_verification",
        "method": "deterministic",
        "langchain_used": False,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "thresholds": {
            "semantic_similarity": SEMANTIC_THRESHOLD,
            "token_overlap": TOKEN_OVERLAP_THRESHOLD,
            "support_score": SUPPORT_SCORE_THRESHOLD,
        },
        "evidence": {
            "core_chunks": len(core_chunks),
            "patient_chunks": len(patient_chunks),
            "total_chunks": (
                len(core_chunks)
                + len(patient_chunks)
            ),
        },
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "test_accuracy": round(
                test_accuracy,
                2,
            ),
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "partially_supported_claims": partial_claims,
            "unsupported_claims": unsupported_claims,
            "claim_support_rate": round(
                claim_support_rate,
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
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    write_report(
        results
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Total tests: "
        f"{total_tests}"
    )

    print(
        f"Passed: "
        f"{passed_tests}"
    )

    print(
        f"Failed: "
        f"{failed_tests}"
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print()

    print(
        f"Total claims: "
        f"{total_claims}"
    )

    print(
        f"Supported claims: "
        f"{supported_claims}"
    )

    print(
        f"Partially supported claims: "
        f"{partial_claims}"
    )

    print(
        f"Unsupported claims: "
        f"{unsupported_claims}"
    )

    print(
        f"Claim support rate: "
        f"{claim_support_rate:.2f}%"
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
        "Claim Verification tests completed."
    )


if __name__ == "__main__":
    main()
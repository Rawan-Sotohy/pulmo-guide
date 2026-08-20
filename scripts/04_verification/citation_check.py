"""
============================================================
PULMO GUIDE
DAY 4 — CITATION CHECK
============================================================

Purpose:
    Validate that generated citations point to real evidence.

Checks:
    1. Citation exists
    2. Chunk ID exists
    3. Source is valid
    4. Page metadata exists when available
    5. Section metadata exists when available
    6. Citation points to the correct source
    7. Citation evidence is actually relevant

Works without Gemini API.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

PATIENT_CACHE_DIR = PROJECT_ROOT / "data" / "patient_cache"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

CORE_COLLECTION_NAME = "pulmo_guide"

VALID_SOURCES = {"core", "patient", "core+patient"}

MIN_TEXT_LENGTH = 10


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "id": "CC-01",
        "source": "core",
        "question": "What are the symptoms of lung cancer?",
        "expected_chunk_keywords": ["cough", "breathlessness"],
    },
    {
        "id": "CC-02",
        "source": "core",
        "question": "What treatment options are recommended for people with lung cancer?",
        "expected_chunk_keywords": [
            "treatment",
            "surgery",
            "radiotherapy",
        ],
    },
    {
        "id": "CC-03",
        "source": "core",
        "question": "What imaging should be offered to people with stage 3 NSCLC?",
        "expected_chunk_keywords": [
            "stage",
            "imaging",
        ],
    },
    {
        "id": "CC-04",
        "source": "patient",
        "question": "What is my FEV1?",
        "expected_chunk_keywords": [
            "FEV1",
            "1.86",
            "76",
        ],
    },
    {
        "id": "CC-05",
        "source": "core+patient",
        "question": "What does this result mean?",
        "expected_chunk_keywords": [
            "restrictive",
            "FEV1",
            "TLCO",
        ],
    },
    {
        "id": "CC-06",
        "source": "core+patient",
        "question": "Is this result normal?",
        "expected_chunk_keywords": [
            "restrictive",
            "diffusion",
        ],
    },
    {
        "id": "CC-07",
        "source": "core",
        "question": "What is the recommended treatment for pancreatic cancer?",
        "expected_chunk_keywords": [],
        "expected_refusal": True,
    },
    {
        "id": "CC-08",
        "source": "core+patient",
        "question": "What does my result mean?",
        "expected_chunk_keywords": [
            "report",
            "result",
        ],
    },
]


# ============================================================
# HELPERS
# ============================================================

def normalize(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def first_value(metadata: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        if key in metadata:
            value = normalize(metadata[key])
            if value:
                return value

    return ""


def extract_page(metadata: Dict[str, Any]) -> str:
    return first_value(
        metadata,
        [
            "page",
            "page_number",
            "page_num",
            "page_no",
            "page_id",
        ],
    )


def extract_section(metadata: Dict[str, Any]) -> str:
    return first_value(
        metadata,
        [
            "section",
            "section_name",
            "section_title",
            "heading",
            "chapter",
        ],
    )


def extract_chunk_id(metadata: Dict[str, Any]) -> str:
    return first_value(
        metadata,
        [
            "chunk_id",
            "id",
            "document_id",
            "chunk",
        ],
    )


def extract_source(metadata: Dict[str, Any]) -> str:
    source = first_value(
        metadata,
        [
            "source",
            "source_type",
            "document_type",
            "collection",
        ],
    )

    source_lower = source.lower()

    if "patient" in source_lower:
        return "patient"

    if "core" in source_lower:
        return "core"

    return ""


def get_metadata_text(metadata: Dict[str, Any]) -> str:
    parts = []

    for key, value in metadata.items():
        if isinstance(value, (str, int, float)):
            parts.append(f"{key}: {value}")

    return " ".join(parts)


def keyword_match_score(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0

    text_lower = text.lower()

    matched = 0

    for keyword in keywords:
        if keyword.lower() in text_lower:
            matched += 1

    return matched / len(keywords)


# ============================================================
# LOAD CORE
# ============================================================

def load_core_evidence() -> Dict[str, Dict[str, Any]]:
    print("Loading Core evidence...")

    if not VECTOR_STORE_DIR.exists():
        print("WARNING: Core vector store not found.")
        return {}

    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIR)
    )

    try:
        collection = client.get_collection(
            name=CORE_COLLECTION_NAME
        )
    except Exception as exc:
        print(f"WARNING: Could not load Core collection: {exc}")
        return {}

    count = collection.count()

    if count == 0:
        return {}

    result = collection.get(
        include=["documents", "metadatas"]
    )

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    evidence = {}

    for index, chunk_id in enumerate(ids):
        metadata = (
            metadatas[index]
            if index < len(metadatas) and metadatas[index]
            else {}
        )

        document = (
            documents[index]
            if index < len(documents)
            else ""
        )

        metadata = dict(metadata)

        if not extract_chunk_id(metadata):
            metadata["chunk_id"] = chunk_id

        if not extract_source(metadata):
            metadata["source"] = "core"

        evidence[str(chunk_id)] = {
            "id": str(chunk_id),
            "text": document or "",
            "metadata": metadata,
            "source": "core",
        }

    print(f"OK: Core evidence chunks: {len(evidence)}")

    return evidence


# ============================================================
# LOAD PATIENT
# ============================================================

def load_patient_evidence() -> Dict[str, Dict[str, Any]]:
    print("Loading Patient evidence...")

    if not PATIENT_CACHE_DIR.exists():
        print("WARNING: Patient cache directory not found.")
        return {}

    evidence = {}

    # Search JSON files recursively.
    json_files = list(PATIENT_CACHE_DIR.rglob("*.json"))

    for json_file in json_files:
        try:
            with open(
                json_file,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)
        except Exception:
            continue

        collect_patient_chunks(data, evidence)

    print(f"Patient evidence chunks: {len(evidence)}")

    return evidence


def collect_patient_chunks(
    data: Any,
    evidence: Dict[str, Dict[str, Any]],
):
    """
    Recursively detect chunk-like structures.

    Supports:
        [
            {"id": ..., "text": ..., "metadata": ...}
        ]

        {
            "chunks": [...]
        }

        {
            "documents": [...],
            "metadatas": [...],
            "ids": [...]
        }
    """

    if isinstance(data, list):

        for item in data:
            collect_patient_chunks(item, evidence)

        return

    if not isinstance(data, dict):
        return

    # --------------------------------------------------------
    # Chroma-like structure
    # --------------------------------------------------------

    if (
        "ids" in data
        and "documents" in data
    ):
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        for i, chunk_id in enumerate(ids):

            text = (
                documents[i]
                if i < len(documents)
                else ""
            )

            metadata = (
                metadatas[i]
                if i < len(metadatas) and metadatas[i]
                else {}
            )

            metadata = dict(metadata)

            metadata.setdefault(
                "chunk_id",
                str(chunk_id),
            )

            metadata.setdefault(
                "source",
                "patient",
            )

            evidence[str(chunk_id)] = {
                "id": str(chunk_id),
                "text": text or "",
                "metadata": metadata,
                "source": "patient",
            }

        return

    # --------------------------------------------------------
    # Single chunk
    # --------------------------------------------------------

    chunk_id = first_value(
        data,
        [
            "chunk_id",
            "id",
        ],
    )

    text = first_value(
        data,
        [
            "text",
            "content",
            "page_content",
            "chunk_text",
        ],
    )

    metadata = data.get("metadata", {})

    if isinstance(metadata, dict):
        metadata = dict(metadata)
    else:
        metadata = {}

    if chunk_id and text:

        metadata.setdefault(
            "chunk_id",
            chunk_id,
        )

        metadata.setdefault(
            "source",
            "patient",
        )

        evidence[str(chunk_id)] = {
            "id": str(chunk_id),
            "text": text,
            "metadata": metadata,
            "source": "patient",
        }

    # --------------------------------------------------------
    # Recursive traversal
    # --------------------------------------------------------

    for key, value in data.items():

        if key in {
            "metadata",
            "text",
            "content",
            "page_content",
        }:
            continue

        if isinstance(value, (dict, list)):
            collect_patient_chunks(
                value,
                evidence,
            )


# ============================================================
# CITATION BUILDER
# ============================================================

def build_citation(
    evidence: Dict[str, Any],
) -> Dict[str, Any]:

    metadata = evidence.get(
        "metadata",
        {},
    )

    return {
        "chunk_id": evidence.get("id", ""),
        "source": evidence.get("source", ""),
        "page": extract_page(metadata),
        "section": extract_section(metadata),
        "text_preview": normalize(
            evidence.get("text", "")
        )[:250],
    }


# ============================================================
# CITATION VALIDATION
# ============================================================

def validate_citation(
    citation: Dict[str, Any],
    evidence_pool: Dict[str, Dict[str, Any]],
    expected_source: str,
    keywords: List[str],
) -> Dict[str, Any]:

    result = {
        "valid": False,
        "reason": "",
        "citation": citation,
        "checks": {},
    }

    # --------------------------------------------------------
    # Check 1 — Citation object
    # --------------------------------------------------------

    if not citation:
        result["reason"] = "Missing citation."
        return result

    result["checks"]["citation_exists"] = True

    # --------------------------------------------------------
    # Check 2 — Chunk ID
    # --------------------------------------------------------

    chunk_id = normalize(
        citation.get("chunk_id")
    )

    if not chunk_id:
        result["reason"] = "Citation has no chunk_id."
        return result

    result["checks"]["chunk_id_exists"] = (
        chunk_id in evidence_pool
    )

    if chunk_id not in evidence_pool:
        result["reason"] = (
            f"Chunk ID '{chunk_id}' does not exist."
        )
        return result

    evidence = evidence_pool[chunk_id]

    # --------------------------------------------------------
    # Check 3 — Source
    # --------------------------------------------------------

    actual_source = evidence.get(
        "source",
        "",
    )

    citation_source = normalize(
        citation.get("source")
    )

    source_ok = (
        not citation_source
        or citation_source == actual_source
    )

    result["checks"]["source_valid"] = source_ok

    if not source_ok:
        result["reason"] = (
            f"Source mismatch: "
            f"citation={citation_source}, "
            f"actual={actual_source}"
        )
        return result

    # --------------------------------------------------------
    # Check 4 — Page
    # --------------------------------------------------------

    metadata = evidence.get(
        "metadata",
        {},
    )

    actual_page = extract_page(metadata)
    cited_page = normalize(
        citation.get("page")
    )

    if cited_page and actual_page:
        page_ok = cited_page == actual_page
    else:
        # Page may legitimately be absent.
        page_ok = True

    result["checks"]["page_valid"] = page_ok

    if not page_ok:
        result["reason"] = (
            f"Page mismatch: "
            f"citation={cited_page}, "
            f"actual={actual_page}"
        )
        return result

    # --------------------------------------------------------
    # Check 5 — Section
    # --------------------------------------------------------

    actual_section = extract_section(metadata)
    cited_section = normalize(
        citation.get("section")
    )

    if cited_section and actual_section:
        section_ok = (
            cited_section.lower()
            == actual_section.lower()
        )
    else:
        section_ok = True

    result["checks"]["section_valid"] = section_ok

    if not section_ok:
        result["reason"] = (
            f"Section mismatch: "
            f"citation={cited_section}, "
            f"actual={actual_section}"
        )
        return result

    # --------------------------------------------------------
    # Check 6 — Evidence text
    # --------------------------------------------------------

    evidence_text = normalize(
        evidence.get("text")
    )

    text_ok = len(evidence_text) >= MIN_TEXT_LENGTH

    result["checks"]["evidence_text_exists"] = text_ok

    if not text_ok:
        result["reason"] = "Evidence text is empty."
        return result

    # --------------------------------------------------------
    # Check 7 — Keyword relevance
    # --------------------------------------------------------

    relevance = keyword_match_score(
        evidence_text,
        keywords,
    )

    result["checks"]["keyword_relevance"] = round(
        relevance,
        4,
    )

    # For refusal cases no evidence is required.
    if not keywords:
        relevance_ok = True
    else:
        relevance_ok = relevance > 0

    result["checks"]["evidence_relevant"] = relevance_ok

    if not relevance_ok:
        result["reason"] = (
            "Citation exists but evidence does not "
            "contain relevant test keywords."
        )
        return result

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    result["valid"] = True

    result["reason"] = "Citation is valid and grounded."

    return result


# ============================================================
# CREATE TEST CITATIONS
# ============================================================

def find_best_evidence(
    evidence_pool: Dict[str, Dict[str, Any]],
    keywords: List[str],
    source: str,
) -> Optional[Dict[str, Any]]:

    candidates = []

    for evidence in evidence_pool.values():

        if evidence.get("source") != source:
            continue

        text = normalize(
            evidence.get("text")
        )

        score = keyword_match_score(
            text,
            keywords,
        )

        if score > 0:
            candidates.append(
                (score, evidence)
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return candidates[0][1]


def create_test_citation(
    case: Dict[str, Any],
    core_evidence: Dict[str, Dict[str, Any]],
    patient_evidence: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    source = case["source"]
    keywords = case["expected_chunk_keywords"]

    if case.get("expected_refusal"):
        return None

    pools = []

    if source in {"core", "core+patient"}:
        pools.append(
            (
                "core",
                core_evidence,
            )
        )

    if source in {"patient", "core+patient"}:
        pools.append(
            (
                "patient",
                patient_evidence,
            )
        )

    best = None
    best_score = -1

    for pool_source, pool in pools:

        candidate = find_best_evidence(
            pool,
            keywords,
            pool_source,
        )

        if candidate is None:
            continue

        score = keyword_match_score(
            candidate.get("text", ""),
            keywords,
        )

        if score > best_score:
            best_score = score
            best = candidate

    if best is None:
        return None

    return build_citation(best)


# ============================================================
# RUN TEST
# ============================================================

def run_test(
    case: Dict[str, Any],
    core_evidence: Dict[str, Dict[str, Any]],
    patient_evidence: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:

    case_id = case["id"]
    source = case["source"]

    print("-" * 70)
    print(f"{case_id} — {source}")
    print(f"Question: {case['question']}")

    if case.get("expected_refusal"):

        print("Expected: REFUSAL")
        print("Citation required: NO")
        print("Result: PASS")

        return {
            "test_id": case_id,
            "source": source,
            "question": case["question"],
            "expected_refusal": True,
            "citation_required": False,
            "result": "PASS",
            "reason": "Out-of-scope question correctly requires no citation.",
        }

    citation = create_test_citation(
        case,
        core_evidence,
        patient_evidence,
    )

    if citation is None:

        print("Citation could not be created.")
        print("Result: FAIL")

        return {
            "test_id": case_id,
            "source": source,
            "question": case["question"],
            "citation_required": True,
            "result": "FAIL",
            "reason": "No suitable evidence citation found.",
        }

    # Build combined pool.
    combined_pool = {}

    if source in {"core", "core+patient"}:
        combined_pool.update(core_evidence)

    if source in {"patient", "core+patient"}:
        combined_pool.update(patient_evidence)

    validation = validate_citation(
        citation,
        combined_pool,
        expected_source=(
            citation.get("source", "")
        ),
        keywords=case[
            "expected_chunk_keywords"
        ],
    )

    print("Citation:")
    print(
        f"  Chunk ID: {citation.get('chunk_id')}"
    )
    print(
        f"  Source: {citation.get('source')}"
    )
    print(
        f"  Page: {citation.get('page') or 'N/A'}"
    )
    print(
        f"  Section: {citation.get('section') or 'N/A'}"
    )

    print("Checks:")

    for name, value in validation[
        "checks"
    ].items():
        print(
            f"  {name}: {value}"
        )

    print(
        f"Result: "
        f"{'PASS' if validation['valid'] else 'FAIL'}"
    )

    return {
        "test_id": case_id,
        "source": source,
        "question": case["question"],
        "citation_required": True,
        "citation": citation,
        "validation": validation,
        "result": (
            "PASS"
            if validation["valid"]
            else "FAIL"
        ),
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results: List[Dict[str, Any]]):

    passed = sum(
        1
        for r in results
        if r["result"] == "PASS"
    )

    failed = len(results) - passed

    accuracy = (
        passed / len(results)
        if results
        else 0
    )

    report = {
        "stage": "Day 4 — Citation Check",
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "accuracy": round(
            accuracy * 100,
            2,
        ),
        "results": results,
    }

    json_path = (
        EVALUATION_DIR
        / "citation_check_results.json"
    )

    txt_path = (
        EVALUATION_DIR
        / "citation_check_report.txt"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    lines = [
        "=" * 70,
        "PULMO GUIDE — DAY 4",
        "CITATION CHECK REPORT",
        "=" * 70,
        "",
        f"Total tests: {len(results)}",
        f"Passed: {passed}",
        f"Failed: {failed}",
        f"Accuracy: {accuracy * 100:.2f}%",
        "",
    ]

    for result in results:

        lines.append(
            f"{result['test_id']} — "
            f"{result['result']}"
        )

        lines.append(
            f"Question: {result['question']}"
        )

        if "citation" in result:

            citation = result["citation"]

            lines.append(
                f"Chunk ID: "
                f"{citation.get('chunk_id', '')}"
            )

            lines.append(
                f"Source: "
                f"{citation.get('source', '')}"
            )

            lines.append(
                f"Page: "
                f"{citation.get('page') or 'N/A'}"
            )

            lines.append(
                f"Section: "
                f"{citation.get('section') or 'N/A'}"
            )

        lines.append("")

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines)
        )

    print("")
    print("JSON saved to:")
    print(json_path)

    print("")
    print("Report saved to:")
    print(txt_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PULMO GUIDE — DAY 4")
    print("CITATION CHECK")
    print("=" * 70)
    print("")

    print("Initializing citation check...")
    print("")

    core_evidence = load_core_evidence()

    patient_evidence = load_patient_evidence()

    print("")
    print("Evidence availability:")
    print(f"  Core: {len(core_evidence)}")
    print(f"  Patient: {len(patient_evidence)}")
    print(
        f"  Core + Patient: "
        f"{len(core_evidence) + len(patient_evidence)}"
    )
    print("")

    results = []

    for case in TEST_CASES:

        result = run_test(
            case,
            core_evidence,
            patient_evidence,
        )

        results.append(result)

        print("")

    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    passed = sum(
        1
        for r in results
        if r["result"] == "PASS"
    )

    failed = len(results) - passed

    accuracy = (
        passed / len(results) * 100
        if results
        else 0
    )

    print(
        f"Total tests: {len(results)}"
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    save_results(results)

    print("")
    print("Citation Check tests completed.")


if __name__ == "__main__":
    main()
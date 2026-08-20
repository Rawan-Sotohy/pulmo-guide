"""
============================================================
PULMO GUIDE — DAY 4
AUTOMATIC CLAIM EXTRACTION
============================================================

Purpose
-------
Extract factual claims from generated / grounded answers.

Project-specific design
-----------------------
- Supports CORE and PATIENT evidence modes.
- Does NOT rebuild retrieval.
- Reuses the existing Confidence Gate pipeline when available.
- Uses LLM extraction when available.
- Falls back to deterministic sentence/claim extraction when
  LLM quota is unavailable.
- Never invents claims.
- Saves JSON + human-readable report.

Run from project root:

    python scripts/04_verification/claim_extraction.py
============================================================
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERIFICATION_DIR = PROJECT_ROOT / "scripts" / "04_verification"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))


# ============================================================
# OUTPUT FILES
# ============================================================

JSON_OUTPUT = (
    EVALUATION_DIR /
    "claim_extraction_results.json"
)

REPORT_OUTPUT = (
    EVALUATION_DIR /
    "claim_extraction_report.txt"
)


# ============================================================
# INPUT
# ============================================================

CONFIDENCE_RESULTS = (
    EVALUATION_DIR /
    "confidence_gate_results.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_CLAIM_WORDS = 5
MAX_CLAIMS_PER_ANSWER = 20

# The LLM is preferred when available.
USE_LLM = True

# Model used only for claim extraction.
# This is intentionally separate from generation logic.
LLM_MODEL = os.getenv(
    "PULMO_CLAIM_MODEL",
    "gemini-2.5-flash"
)


# ============================================================
# TEST CASES
# ============================================================
#
# These are representative of YOUR actual Core + Patient
# pipeline.
#
# We deliberately include:
#   - Core answers
#   - Patient answers
#   - Core + Patient answers
#   - refusal
#   - cautious/partial answer
#
# If confidence_gate_results.json already contains answers,
# those are preferred automatically.
#

DEFAULT_TEST_CASES = [
    {
        "id": "CE-01",
        "source": "core",
        "question": "What are the symptoms of lung cancer?",
        "answer": (
            "Common symptoms of lung cancer can include a persistent "
            "cough, coughing up blood, chest pain, and breathlessness. "
            "The guideline should be consulted for the complete list "
            "and clinical context."
        ),
    },
    {
        "id": "CE-02",
        "source": "core",
        "question": (
            "What treatment options are recommended for people "
            "with lung cancer?"
        ),
        "answer": (
            "Treatment options depend on the type and stage of lung "
            "cancer and may include surgery, radiotherapy, systemic "
            "anticancer treatment, or combinations of these approaches. "
            "The appropriate option depends on the patient's clinical "
            "assessment."
        ),
    },
    {
        "id": "CE-03",
        "source": "core",
        "question": (
            "What imaging should be offered to people with stage 3 NSCLC?"
        ),
        "answer": (
            "Imaging recommendations for people with stage 3 NSCLC "
            "depend on the clinical situation and the extent of disease. "
            "The relevant guideline recommendations should be followed "
            "for selecting appropriate imaging."
        ),
    },
    {
        "id": "CE-04",
        "source": "patient",
        "question": "What is my FEV1?",
        "answer": (
            "Your report records an FEV1 of 1.86 L, which is 76% of "
            "the predicted value."
        ),
    },
    {
        "id": "CE-05",
        "source": "core+patient",
        "question": "What does this result mean?",
        "answer": (
            "The report describes a mild restrictive ventilatory pattern "
            "with mildly reduced diffusion capacity. The measured FEV1 "
            "is 76% of predicted and TLCO is also 76% of predicted. "
            "The report states that these findings do not preclude "
            "consideration of treatment with curative intent."
        ),
    },
    {
        "id": "CE-06",
        "source": "core+patient",
        "question": "Is this result normal?",
        "answer": (
            "The report does not describe the result as completely normal. "
            "It describes a mild restrictive ventilatory pattern and "
            "mildly reduced diffusion capacity."
        ),
    },
    {
        "id": "CE-07",
        "source": "core",
        "question": "What is the recommended treatment for pancreatic cancer?",
        "answer": (
            "I couldn't answer this question because the indexed "
            "guideline covers lung cancer and does not provide evidence "
            "about the condition mentioned in your question."
        ),
    },
    {
        "id": "CE-08",
        "source": "core+patient",
        "question": "What does my result mean?",
        "answer": (
            "The available report provides findings that can be described "
            "from the document, but interpretation should remain limited "
            "to the supplied evidence."
        ),
    },
]


# ============================================================
# OPTIONAL GEMINI SETUP
# ============================================================

gemini_client = None
gemini_types = None
llm_available = False


def initialize_llm() -> bool:
    """
    Initialize Gemini if an API key and package are available.

    Failure is intentionally non-fatal because the extraction
    pipeline has a deterministic fallback.
    """

    global gemini_client
    global gemini_types
    global llm_available

    if not USE_LLM:
        return False

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("INFO: GEMINI_API_KEY not found.")
        print("Using deterministic claim extraction fallback.")
        return False

    try:
        from google import genai
        from google.genai import types

        gemini_client = genai.Client(api_key=api_key)
        gemini_types = types

        llm_available = True

        print(
            f"OK: Claim extraction LLM ready "
            f"({LLM_MODEL})."
        )

        return True

    except Exception as exc:
        print(
            "WARNING: Could not initialize Gemini.\n"
            f"Reason: {exc}\n"
            "Using deterministic fallback."
        )

        return False


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def clean_text(text: Any) -> str:
    """
    Normalize generated answer text.
    """

    if text is None:
        return ""

    text = str(text)

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# REFUSAL DETECTION
# ============================================================

REFUSAL_PATTERNS = [
    r"\bcouldn't answer\b",
    r"\bcannot answer\b",
    r"\bcan't answer\b",
    r"\bunable to answer\b",
    r"\bnot enough evidence\b",
    r"\binsufficient evidence\b",
    r"\boutside the scope\b",
    r"\bout[- ]of[- ]scope\b",
    r"\bdoes not provide evidence\b",
    r"\bdo not provide evidence\b",
    r"\bnot covered\b",
    r"\bno relevant evidence\b",
]


def is_refusal(answer: str) -> bool:
    """
    Detect a system refusal.

    A refusal is NOT treated as a medical factual claim.
    """

    normalized = answer.lower()

    return any(
        re.search(pattern, normalized)
        for pattern in REFUSAL_PATTERNS
    )


# ============================================================
# NON-CLAIM FILTERS
# ============================================================

NON_CLAIM_PATTERNS = [
    r"^recommendation:?$",
    r"^excerpt:?$",
    r"^citation:?$",
    r"^answer:?$",
    r"^source:?$",
    r"^note:?$",
    r"^disclaimer:?$",
    r"^i (can|cannot|could|couldn't|can't|am) ",
    r"^the (available )?evidence (is|was) ",
    r"^the guideline (is|was) ",
]


def looks_like_non_claim(sentence: str) -> bool:
    """
    Remove obvious headings, meta statements, and assistant
    process language.
    """

    text = sentence.strip()

    if not text:
        return True

    lowered = text.lower()

    for pattern in NON_CLAIM_PATTERNS:
        if re.search(pattern, lowered):
            return True

    return False


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter.

    No additional model is required.
    """

    text = clean_text(text)

    if not text:
        return []

    # Protect common abbreviations / decimals temporarily.
    protected = text

    protected = re.sub(
        r"\b(e\.g|i\.e|etc)\.",
        lambda m: m.group(0).replace(".", "<DOT>"),
        protected,
        flags=re.IGNORECASE,
    )

    protected = re.sub(
        r"(?<=\d)\.(?=\d)",
        "<DECIMAL>",
        protected,
    )

    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        protected,
    )

    sentences = []

    for part in parts:
        part = part.replace("<DOT>", ".")
        part = part.replace("<DECIMAL>", ".")
        part = part.strip()

        if part:
            sentences.append(part)

    return sentences


# ============================================================
# CLAIM CANDIDATE FILTER
# ============================================================

def is_claim_candidate(sentence: str) -> bool:
    """
    Decide whether a sentence is worth treating as a factual claim.
    """

    sentence = clean_text(sentence)

    if len(sentence.split()) < MIN_CLAIM_WORDS:
        return False

    if looks_like_non_claim(sentence):
        return False

    if is_refusal(sentence):
        return False

    # Remove pure markdown headings.
    if re.match(r"^#{1,6}\s+", sentence):
        return False

    # Remove citation-only strings.
    if re.fullmatch(r"\[[^\]]+\]", sentence):
        return False

    return True


# ============================================================
# ATOMIC CLAIM SPLITTING
# ============================================================

CLAIM_SPLIT_WORDS = [
    " and ",
    " while ",
    " whereas ",
]


def split_atomic_claims(sentence: str) -> list[str]:
    """
    Lightweight atomic-claim splitting.

    We do NOT aggressively split every 'and', because medical
    statements often contain connected concepts.

    The function only splits obvious multi-clause constructions.
    """

    sentence = clean_text(sentence)

    if not sentence:
        return []

    # Keep numeric expressions such as:
    # FEV1 76% and TLCO 76%
    # together when both refer to the same reported finding.

    # First split semicolon-separated independent statements.
    parts = re.split(r"\s*;\s*", sentence)

    results = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # Split bullet-like clauses.
        if " — " in part:
            subparts = re.split(r"\s+—\s+", part)
        else:
            subparts = [part]

        for subpart in subparts:
            subpart = clean_text(subpart)

            if is_claim_candidate(subpart):
                results.append(subpart)

    return results


# ============================================================
# DETERMINISTIC EXTRACTION
# ============================================================

def extract_claims_deterministic(answer: str) -> list[str]:
    """
    Reliable local fallback.

    This is intentionally conservative:
    it prefers fewer claims over hallucinated claims.
    """

    answer = clean_text(answer)

    if not answer:
        return []

    if is_refusal(answer):
        return []

    candidates = []

    for sentence in split_sentences(answer):

        if not is_claim_candidate(sentence):
            continue

        atomic = split_atomic_claims(sentence)

        for claim in atomic:
            claim = clean_text(claim)

            if not claim:
                continue

            if claim not in candidates:
                candidates.append(claim)

            if len(candidates) >= MAX_CLAIMS_PER_ANSWER:
                break

        if len(candidates) >= MAX_CLAIMS_PER_ANSWER:
            break

    return candidates


# ============================================================
# LLM PROMPT
# ============================================================

CLAIM_EXTRACTION_SYSTEM_PROMPT = """
You are an automatic factual claim extraction system for a
grounded medical RAG application.

Your task is ONLY to extract factual claims explicitly stated
in the supplied answer.

IMPORTANT RULES:

1. Use ONLY the supplied answer.
2. Do NOT add outside medical knowledge.
3. Do NOT correct the answer.
4. Do NOT interpret the answer.
5. Do NOT infer information that is not explicitly stated.
6. Ignore headings such as Recommendation, Excerpt, Citation,
   Source, Answer, or Note.
7. Ignore citations themselves.
8. Ignore refusal/meta statements.
9. Split complex statements into atomic factual claims when
   doing so does not change their meaning.
10. Preserve numbers, percentages, units, medical terminology,
    conditions, and qualifiers exactly.
11. If the answer contains no factual claims, return an empty list.
12. Return JSON only.

Required JSON format:

{
  "claims": [
    "claim 1",
    "claim 2"
  ]
}
""".strip()


# ============================================================
# LLM EXTRACTION
# ============================================================

def extract_claims_llm(answer: str) -> list[str]:
    """
    Extract claims using Gemini.

    Raises on failure so the caller can use the deterministic
    fallback.
    """

    if not llm_available:
        raise RuntimeError("LLM is not available.")

    response = gemini_client.models.generate_content(
        model=LLM_MODEL,
        contents=answer,
        config=gemini_types.GenerateContentConfig(
            system_instruction=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            temperature=0,
        ),
    )

    raw = clean_text(
        getattr(response, "text", "")
    )

    if not raw:
        raise RuntimeError(
            "LLM returned an empty response."
        )

    # Remove accidental markdown fences.
    raw = re.sub(
        r"^```json\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    raw = re.sub(
        r"^```\s*",
        "",
        raw,
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw,
    )

    data = json.loads(raw)

    claims = data.get("claims", [])

    if not isinstance(claims, list):
        raise ValueError(
            "LLM response does not contain a claims list."
        )

    cleaned_claims = []

    for claim in claims:

        if not isinstance(claim, str):
            continue

        claim = clean_text(claim)

        if not claim:
            continue

        if is_refusal(claim):
            continue

        if looks_like_non_claim(claim):
            continue

        if len(claim.split()) < MIN_CLAIM_WORDS:
            continue

        if claim not in cleaned_claims:
            cleaned_claims.append(claim)

        if len(cleaned_claims) >= MAX_CLAIMS_PER_ANSWER:
            break

    return cleaned_claims


# ============================================================
# HYBRID EXTRACTION
# ============================================================

def extract_claims(
    answer: str,
) -> tuple[list[str], str, str | None]:
    """
    Main extraction function.

    Returns:
        claims
        method
        error
    """

    answer = clean_text(answer)

    if not answer:
        return [], "empty", None

    if is_refusal(answer):
        return [], "refusal", None

    if llm_available:

        try:
            claims = extract_claims_llm(answer)

            # Safety fallback if the LLM unexpectedly returns
            # no claims from a clearly factual answer.
            if claims:
                return claims, "llm", None

            fallback = extract_claims_deterministic(answer)

            if fallback:
                return (
                    fallback,
                    "deterministic_fallback",
                    "LLM returned no claims.",
                )

            return [], "llm_empty", None

        except Exception as exc:

            fallback = extract_claims_deterministic(answer)

            return (
                fallback,
                "deterministic_fallback",
                str(exc),
            )

    claims = extract_claims_deterministic(answer)

    return claims, "deterministic", None


# ============================================================
# LOAD CONFIDENCE GATE RESULTS
# ============================================================

def load_confidence_results() -> list[dict[str, Any]]:
    """
    Load the real Day 4 Confidence Gate output when available.
    """

    if not CONFIDENCE_RESULTS.exists():
        return []

    try:
        with open(
            CONFIDENCE_RESULTS,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        # Support several common output structures.
        if isinstance(data, list):
            return data

        if isinstance(data, dict):

            for key in [
                "results",
                "tests",
                "test_results",
                "evaluation",
                "cases",
            ]:
                value = data.get(key)

                if isinstance(value, list):
                    return value

    except Exception as exc:
        print(
            "WARNING: Could not read confidence gate results:"
        )
        print(exc)

    return []


# ============================================================
# EXTRACT ANSWER FROM CONFIDENCE RESULT
# ============================================================

def get_field(
    item: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:

    for key in keys:
        if key in item:
            return item[key]

    return default


def normalize_test_case(
    item: dict[str, Any],
) -> dict[str, Any]:

    return {
        "id": get_field(
            item,
            "id",
            "test_id",
            "case_id",
            default="UNKNOWN",
        ),
        "source": get_field(
            item,
            "source",
            "source_mode",
            "mode",
            default="unknown",
        ),
        "question": get_field(
            item,
            "question",
            "query",
            default="",
        ),
        "answer": get_field(
            item,
            "answer",
            "generated_answer",
            "response",
            default="",
        ),
        "status": get_field(
            item,
            "status",
            "result",
            default="unknown",
        ),
        "confidence": get_field(
            item,
            "confidence",
            "confidence_score",
            "score",
            default=None,
        ),
    }


# ============================================================
# BUILD TEST SET
# ============================================================

def build_test_set() -> list[dict[str, Any]]:
    """
    Prefer the actual Confidence Gate answers.

    If the confidence gate JSON does not contain answers,
    use the representative project-specific test cases.
    """

    actual_results = load_confidence_results()

    normalized = []

    for item in actual_results:

        if not isinstance(item, dict):
            continue

        case = normalize_test_case(item)

        if case["question"]:

            # Some confidence gate implementations store
            # grounded fallback text under another field.
            if not case["answer"]:

                evidence_answer = get_field(
                    item,
                    "grounded_answer",
                    "fallback_answer",
                    "grounded_response",
                    default="",
                )

                if evidence_answer:
                    case["answer"] = evidence_answer

            normalized.append(case)

    # If real answers are available, use them.
    cases_with_answers = [
        item
        for item in normalized
        if clean_text(item["answer"])
    ]

    if cases_with_answers:
        print(
            f"Loaded {len(cases_with_answers)} real "
            "Confidence Gate answers."
        )

        return cases_with_answers

    print(
        "INFO: Confidence Gate JSON does not contain "
        "usable answer text."
    )

    print(
        "Using project-specific extraction test cases."
    )

    return DEFAULT_TEST_CASES


# ============================================================
# PROCESS ONE TEST
# ============================================================

def process_test_case(
    item: dict[str, Any],
) -> dict[str, Any]:

    case_id = item.get("id", "UNKNOWN")
    source = item.get("source", "unknown")
    question = clean_text(item.get("question", ""))
    answer = clean_text(item.get("answer", ""))

    claims, method, error = extract_claims(answer)

    return {
        "test_id": case_id,
        "source": source,
        "question": question,
        "answer": answer,
        "status": item.get("status", "unknown"),
        "confidence": item.get("confidence"),
        "extraction_method": method,
        "claim_count": len(claims),
        "claims": [
            {
                "claim_id": f"{case_id}-CLM-{index:02d}",
                "text": claim,
            }
            for index, claim in enumerate(
                claims,
                start=1,
            )
        ],
        "error": error,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_result(
    result: dict[str, Any],
) -> tuple[bool, list[str]]:

    errors = []

    answer = result["answer"]
    claims = result["claims"]

    if not answer:
        if claims:
            errors.append(
                "Claims were extracted from an empty answer."
            )

        return len(errors) == 0, errors

    if is_refusal(answer):

        if claims:
            errors.append(
                "Refusal answer incorrectly produced factual claims."
            )

    for claim in claims:

        text = claim.get("text", "")

        if not text:
            errors.append(
                f"Empty claim: {claim.get('claim_id')}"
            )

        if len(text.split()) < MIN_CLAIM_WORDS:
            errors.append(
                f"Claim too short: {claim.get('claim_id')}"
            )

        if is_refusal(text):
            errors.append(
                f"Refusal incorrectly extracted as claim: "
                f"{claim.get('claim_id')}"
            )

    # IDs must be unique.
    ids = [
        claim.get("claim_id")
        for claim in claims
    ]

    if len(ids) != len(set(ids)):
        errors.append(
            "Duplicate claim IDs detected."
        )

    return len(errors) == 0, errors


# ============================================================
# REPORT GENERATION
# ============================================================

def build_report(
    results: list[dict[str, Any]],
    llm_status: bool,
) -> str:

    total_tests = len(results)

    valid_tests = 0

    total_claims = 0

    llm_count = 0
    fallback_count = 0
    refusal_count = 0
    empty_count = 0

    for result in results:

        valid, _ = validate_result(result)

        if valid:
            valid_tests += 1

        total_claims += result["claim_count"]

        method = result["extraction_method"]

        if method == "llm":
            llm_count += 1

        elif method in {
            "deterministic",
            "deterministic_fallback",
        }:
            fallback_count += 1

        elif method == "refusal":
            refusal_count += 1

        elif method in {
            "empty",
            "llm_empty",
        }:
            empty_count += 1

    accuracy = (
        valid_tests / total_tests
        if total_tests
        else 0.0
    )

    lines = []

    lines.append("=" * 70)
    lines.append("PULMO GUIDE — DAY 4")
    lines.append("AUTOMATIC CLAIM EXTRACTION")
    lines.append("=" * 70)
    lines.append("")

    lines.append(
        f"Generated at: "
        f"{datetime.now(timezone.utc).isoformat()}"
    )
    lines.append(
        f"LLM available: {llm_status}"
    )
    lines.append(
        f"LLM model: {LLM_MODEL}"
    )
    lines.append("")

    lines.append("-" * 70)
    lines.append("SUMMARY")
    lines.append("-" * 70)

    lines.append(
        f"Total tests: {total_tests}"
    )
    lines.append(
        f"Valid tests: {valid_tests}"
    )
    lines.append(
        f"Failed tests: {total_tests - valid_tests}"
    )
    lines.append(
        f"Validation accuracy: {accuracy * 100:.2f}%"
    )
    lines.append(
        f"Total extracted claims: {total_claims}"
    )
    lines.append(
        f"LLM extraction: {llm_count}"
    )
    lines.append(
        f"Deterministic/fallback extraction: {fallback_count}"
    )
    lines.append(
        f"Refusal cases: {refusal_count}"
    )
    lines.append(
        f"Empty cases: {empty_count}"
    )
    lines.append("")

    for result in results:

        lines.append("-" * 70)
        lines.append(
            f"{result['test_id']} — "
            f"{result['source']}"
        )
        lines.append(
            f"Question: {result['question']}"
        )
        lines.append(
            f"Method: {result['extraction_method']}"
        )
        lines.append(
            f"Claims: {result['claim_count']}"
        )

        if result["error"]:
            lines.append(
                f"Fallback reason: {result['error']}"
            )

        if result["claims"]:

            for claim in result["claims"]:
                lines.append(
                    f"  {claim['claim_id']}: "
                    f"{claim['text']}"
                )

        else:
            lines.append(
                "  No factual claims extracted."
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append("CLAIM EXTRACTION TEST COMPLETED")
    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    results: list[dict[str, Any]],
) -> None:

    payload = {
        "project": "Pulmo Guide",
        "day": 4,
        "component": "automatic_claim_extraction",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration": {
            "min_claim_words": MIN_CLAIM_WORDS,
            "max_claims_per_answer": MAX_CLAIMS_PER_ANSWER,
            "llm_enabled": USE_LLM,
            "llm_model": LLM_MODEL,
            "fallback_enabled": True,
            "supports_core": True,
            "supports_patient": True,
            "supports_core_patient": True,
        },
        "results": results,
    }

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
    )

    print("=" * 70)
    print("PULMO GUIDE — DAY 4")
    print("AUTOMATIC CLAIM EXTRACTION")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    print("Initializing claim extraction...")

    initialize_llm()

    print()

    # --------------------------------------------------------
    # TEST SET
    # --------------------------------------------------------

    test_cases = build_test_set()

    print(
        f"Total extraction tests: {len(test_cases)}"
    )
    print()

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    results = []

    for item in test_cases:

        case_id = item.get("id", "UNKNOWN")

        print("-" * 70)
        print(
            f"{case_id} — "
            f"{item.get('source', 'unknown')}"
        )
        print(
            f"Question: "
            f"{item.get('question', '')}"
        )

        result = process_test_case(item)

        results.append(result)

        print(
            f"Method: "
            f"{result['extraction_method']}"
        )

        print(
            f"Claims extracted: "
            f"{result['claim_count']}"
        )

        if result["claims"]:

            for claim in result["claims"]:

                print(
                    f"  {claim['claim_id']}: "
                    f"{claim['text']}"
                )

        else:
            print(
                "  No factual claims extracted."
            )

        if result["error"]:
            print(
                f"Fallback/Error: "
                f"{result['error']}"
            )

        print()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    valid_count = 0
    failed_count = 0

    for result in results:

        valid, errors = validate_result(result)

        if valid:
            valid_count += 1
        else:
            failed_count += 1

            print(
                f"VALIDATION FAILURE — "
                f"{result['test_id']}"
            )

            for error in errors:
                print(
                    f"  - {error}"
                )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_json(results)

    report = build_report(
        results,
        llm_available,
    )

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    total = len(results)

    accuracy = (
        valid_count / total * 100
        if total
        else 0.0
    )

    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Total tests: {total}"
    )

    print(
        f"Passed: {valid_count}"
    )

    print(
        f"Failed: {failed_count}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print()

    print("JSON saved to:")
    print(JSON_OUTPUT)

    print()

    print("Report saved to:")
    print(REPORT_OUTPUT)

    print()
    print(
        "Automatic Claim Extraction tests completed."
    )


if __name__ == "__main__":
    main()

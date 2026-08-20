"""
============================================================
PULMO GUIDE
DAY 4 — CONFIDENCE GATE
============================================================

Purpose:
    Guard the RAG generation pipeline by measuring whether
    retrieved evidence is strong enough to support an answer.

Supports:
    - CORE / NICE NG122
    - PATIENT uploaded reports
    - CORE + PATIENT
    - Refusal cases
    - Weak / insufficient evidence cases

Outputs:
    data/evaluation/confidence_gate_results.json
    data/evaluation/confidence_gate_report.txt

Run:
    python scripts/04_verification/confidence_gate.py
============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import sys
import json
import math
import traceback


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENERATION_DIR = (
    PROJECT_ROOT / "scripts" / "03_generation"
)

VERIFICATION_DIR = (
    PROJECT_ROOT / "scripts" / "04_verification"
)

EVALUATION_DIR = (
    PROJECT_ROOT / "data" / "evaluation"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

for directory in [
    GENERATION_DIR,
    VERIFICATION_DIR,
]:
    directory_string = str(directory)

    if directory_string not in sys.path:
        sys.path.append(directory_string)


# ============================================================
# IMPORT REAL PIPELINE
# ============================================================

try:
    from pipeline import run_pipeline
except Exception as error:
    print("\nERROR: Could not import the real generation pipeline.")
    print(error)
    print("\nExpected:")
    print(
        PROJECT_ROOT
        / "scripts"
        / "03_generation"
        / "pipeline.py"
    )
    raise


# ============================================================
# OUTPUT FILES
# ============================================================

JSON_OUTPUT = (
    EVALUATION_DIR
    / "confidence_gate_results.json"
)

REPORT_OUTPUT = (
    EVALUATION_DIR
    / "confidence_gate_report.txt"
)


# ============================================================
# PATIENT PDF
# ============================================================

PATIENT_PDF = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "patient"
    / "pulmonary_function_report.pdf"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum confidence required to allow generation.
CONFIDENCE_THRESHOLD = 0.55

# Evidence score weights.
TOP_SCORE_WEIGHT = 0.50
AVERAGE_SCORE_WEIGHT = 0.30
EVIDENCE_LEVEL_WEIGHT = 0.20

# Retrieval scores can be different between Core and Patient.
# We normalize them safely before using them.

CORE_TOP_K = 5
PATIENT_TOP_K = 5


# ============================================================
# SOURCE TYPES
# ============================================================

SOURCE_CORE = "core"
SOURCE_PATIENT = "patient"
SOURCE_BOTH = "core+patient"


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "id": "CG-01",
        "name": "Core guideline question",
        "query": (
            "What are the symptoms of lung cancer?"
        ),
        "type": "core",
        "expected_source": SOURCE_CORE,
        "should_have_evidence": True,
    },
    {
        "id": "CG-02",
        "name": "Core treatment question",
        "query": (
            "What treatment options are recommended "
            "for people with lung cancer?"
        ),
        "type": "core",
        "expected_source": SOURCE_CORE,
        "should_have_evidence": True,
    },
    {
        "id": "CG-03",
        "name": "Core staging question",
        "query": (
            "What imaging should be offered to people "
            "with stage 3 NSCLC?"
        ),
        "type": "core",
        "expected_source": SOURCE_CORE,
        "should_have_evidence": True,
    },
    {
        "id": "CG-04",
        "name": "Patient FEV1 value",
        "query": (
            "What is my FEV1?"
        ),
        "type": "patient",
        "expected_source": SOURCE_BOTH,
        "should_have_evidence": True,
    },
    {
        "id": "CG-05",
        "name": "Patient follow-up",
        "query": (
            "What does this result mean?"
        ),
        "type": "patient",
        "expected_source": SOURCE_BOTH,
        "should_have_evidence": True,
    },
    {
        "id": "CG-06",
        "name": "Patient result interpretation",
        "query": (
            "Is this result normal?"
        ),
        "type": "patient",
        "expected_source": SOURCE_BOTH,
        "should_have_evidence": True,
    },
    {
        "id": "CG-07",
        "name": "Explicit out-of-scope",
        "query": (
            "What is the recommended treatment "
            "for pancreatic cancer?"
        ),
        "type": "refusal",
        "expected_source": None,
        "should_have_evidence": False,
    },
    {
        "id": "CG-08",
        "name": "Unsupported topic",
        "query": (
            "What is the recommended treatment "
            "for pancreatic cancer according to this guideline?"
        ),
        "type": "refusal",
        "expected_source": None,
        "should_have_evidence": False,
    },
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0
) -> float:

    try:
        number = float(value)

        if math.isnan(number):
            return default

        if math.isinf(number):
            return default

        return number

    except Exception:
        return default


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0
) -> float:

    return max(
        minimum,
        min(maximum, value)
    )


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_score(
    score: float
) -> float:
    """
    Converts common retrieval score ranges into 0..1.

    Handles:
        - cosine similarity [-1, 1]
        - normalized scores [0, 1]
        - BM25 positive scores
        - already normalized hybrid scores
    """

    score = safe_float(score)

    if 0.0 <= score <= 1.0:
        return score

    if -1.0 <= score < 0.0:
        return clamp(
            (score + 1.0) / 2.0
        )

    # Positive scores above 1 are not directly comparable.
    # Compress them using a stable sigmoid-like transformation.
    return clamp(
        score / (score + 1.0)
    )


# ============================================================
# EXTRACT RESULTS
# ============================================================

def get_retrieved_results(
    pipeline_result: Dict[str, Any]
) -> List[Dict[str, Any]]:

    results = pipeline_result.get(
        "generation_results"
    )

    if not results:
        results = pipeline_result.get(
            "retrieved_results"
        )

    if not results:
        results = (
            pipeline_result.get("core_results", [])
            +
            pipeline_result.get("patient_results", [])
        )

    if not isinstance(results, list):
        return []

    return [
        result
        for result in results
        if isinstance(result, dict)
    ]


# ============================================================
# RESULT SCORE
# ============================================================

def get_result_score(
    result: Dict[str, Any]
) -> float:
    """
    Prefer hybrid score.

    Fallback:
        semantic_normalized
        semantic_score
        bm25_normalized
        bm25_score
    """

    possible_scores = [
        result.get("hybrid_score"),
        result.get("semantic_normalized"),
        result.get("semantic_score"),
        result.get("bm25_normalized"),
        result.get("bm25_score"),
    ]

    for score in possible_scores:

        if score is not None:
            return normalize_score(score)

    return 0.0


# ============================================================
# EVIDENCE LEVEL SCORE
# ============================================================

def evidence_level_score(
    pipeline_result: Dict[str, Any]
) -> float:

    evidence = pipeline_result.get(
        "evidence"
    )

    if not isinstance(evidence, dict):
        return 0.5

    level = str(
        evidence.get(
            "evidence_level",
            ""
        )
    ).lower()

    decision = str(
        evidence.get(
            "decision",
            ""
        )
    ).lower()

    if decision == "insufficient":
        return 0.0

    mapping = {
        "high": 1.0,
        "strong": 1.0,
        "medium": 0.70,
        "moderate": 0.70,
        "low": 0.35,
        "weak": 0.25,
    }

    if level in mapping:
        return mapping[level]

    if decision in {
        "sufficient",
        "allow",
        "allowed",
        "pass",
    }:
        return 0.70

    return 0.50


# ============================================================
# CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence(
    pipeline_result: Dict[str, Any]
) -> Dict[str, Any]:

    results = get_retrieved_results(
        pipeline_result
    )

    if not results:

        return {
            "confidence": 0.0,
            "top_score": 0.0,
            "average_score": 0.0,
            "evidence_level_score": 0.0,
            "retrieved_count": 0,
            "gate": "REJECT",
            "reason": "No retrieved evidence.",
        }

    scores = [
        get_result_score(result)
        for result in results
    ]

    scores = [
        clamp(score)
        for score in scores
    ]

    top_score = max(scores)

    average_score = sum(scores) / len(scores)

    evidence_score = evidence_level_score(
        pipeline_result
    )

    confidence = (
        TOP_SCORE_WEIGHT * top_score
        +
        AVERAGE_SCORE_WEIGHT * average_score
        +
        EVIDENCE_LEVEL_WEIGHT * evidence_score
    )

    confidence = clamp(
        confidence
    )

    gate = (
        "PASS"
        if confidence >= CONFIDENCE_THRESHOLD
        else "REJECT"
    )

    reason = (
        "Evidence confidence is above threshold."
        if gate == "PASS"
        else "Evidence confidence is below threshold."
    )

    return {
        "confidence": round(
            confidence,
            4
        ),
        "top_score": round(
            top_score,
            4
        ),
        "average_score": round(
            average_score,
            4
        ),
        "evidence_level_score": round(
            evidence_score,
            4
        ),
        "retrieved_count": len(results),
        "gate": gate,
        "reason": reason,
    }


# ============================================================
# SOURCE VALIDATION
# ============================================================

def validate_source_mode(
    actual_source: Optional[str],
    expected_source: Optional[str]
) -> bool:

    if expected_source is None:
        return actual_source is None

    if actual_source == expected_source:
        return True

    # Patient questions may legitimately route to BOTH.
    if (
        expected_source == SOURCE_BOTH
        and actual_source == SOURCE_PATIENT
    ):
        return True

    return False


# ============================================================
# RUN ONE TEST
# ============================================================

def run_test(
    test_case: Dict[str, Any]
) -> Dict[str, Any]:

    test_id = test_case["id"]
    query = test_case["query"]
    test_type = test_case["type"]

    result: Dict[str, Any] = {
        "test_id": test_id,
        "name": test_case["name"],
        "query": query,
        "type": test_type,
        "expected_source": test_case.get(
            "expected_source"
        ),
        "should_have_evidence": test_case.get(
            "should_have_evidence",
            True
        ),
        "passed": False,
    }

    try:

        patient_pdf = None

        if test_type == "patient":

            if not PATIENT_PDF.exists():

                result.update({
                    "passed": False,
                    "status": "SKIPPED",
                    "reason": (
                        "Patient PDF not found: "
                        f"{PATIENT_PDF}"
                    ),
                })

                return result

            patient_pdf = str(
                PATIENT_PDF
            )

        pipeline_result = run_pipeline(
            query=query,
            patient_pdf=patient_pdf,
            session_id=(
                "day4_confidence_test"
                if patient_pdf
                else None
            ),
        )

        result["pipeline_status"] = (
            pipeline_result.get(
                "status"
            )
        )

        result["pipeline_stage"] = (
            pipeline_result.get(
                "stage"
            )
        )

        result["actual_source"] = (
            pipeline_result.get(
                "source_mode"
            )
        )

        result["answer_available"] = bool(
            str(
                pipeline_result.get(
                    "answer",
                    ""
                )
            ).strip()
        )

        retrieved_results = (
            get_retrieved_results(
                pipeline_result
            )
        )

        result["retrieved_count"] = len(
            retrieved_results
        )

        # ----------------------------------------------------
        # REFUSAL TEST
        # ----------------------------------------------------

        if test_type == "refusal":

            refusal_pass = (
                pipeline_result.get(
                    "status"
                ) == "refused"
            )

            result["gate"] = "REJECT"

            result["confidence"] = 0.0

            result["passed"] = refusal_pass

            result["reason"] = (
                "Correctly refused an out-of-scope "
                "question."
                if refusal_pass
                else
                "Expected refusal but pipeline "
                "did not refuse."
            )

            return result

        # ----------------------------------------------------
        # NORMAL EVIDENCE TEST
        # ----------------------------------------------------

        confidence_data = calculate_confidence(
            pipeline_result
        )

        result["confidence"] = (
            confidence_data["confidence"]
        )

        result["top_score"] = (
            confidence_data["top_score"]
        )

        result["average_score"] = (
            confidence_data["average_score"]
        )

        result["evidence_level_score"] = (
            confidence_data[
                "evidence_level_score"
            ]
        )

        result["gate"] = (
            confidence_data["gate"]
        )

        result["confidence_reason"] = (
            confidence_data["reason"]
        )

        source_pass = validate_source_mode(
            pipeline_result.get(
                "source_mode"
            ),
            test_case.get(
                "expected_source"
            )
        )

        evidence_pass = (
            len(retrieved_results) > 0
        )

        gate_pass = (
            confidence_data["gate"]
            == "PASS"
        )

        expected_evidence = test_case.get(
            "should_have_evidence",
            True
        )

        if expected_evidence:

            result["passed"] = (
                source_pass
                and evidence_pass
                and gate_pass
            )

        else:

            result["passed"] = (
                source_pass
                and not evidence_pass
            )

        result["checks"] = {
            "source_routing": source_pass,
            "evidence_available": evidence_pass,
            "confidence_gate": gate_pass,
        }

        if result["passed"]:

            result["reason"] = (
                "Confidence gate passed."
            )

        else:

            result["reason"] = (
                "One or more confidence gate "
                "conditions failed."
            )

        return result

    except Exception as error:

        result.update({
            "passed": False,
            "status": "ERROR",
            "reason": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        })

        return result


# ============================================================
# RUN ALL TESTS
# ============================================================

def run_all_tests() -> Dict[str, Any]:

    print("=" * 70)
    print("PULMO GUIDE — DAY 4")
    print("CONFIDENCE GATE")
    print("=" * 70)

    print(
        f"\nConfidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )

    results = []

    for test_case in TEST_CASES:

        print("\n" + "-" * 70)

        print(
            f"{test_case['id']} — "
            f"{test_case['name']}"
        )

        print(
            f"Query: {test_case['query']}"
        )

        test_result = run_test(
            test_case
        )

        results.append(
            test_result
        )

        status = (
            "PASS"
            if test_result.get("passed")
            else
            "FAIL"
        )

        if test_result.get(
            "status"
        ) == "SKIPPED":

            status = "SKIPPED"

        print(
            f"Result: {status}"
        )

        if "confidence" in test_result:

            print(
                "Confidence:",
                test_result.get(
                    "confidence"
                )
            )

        if test_result.get(
            "actual_source"
        ):

            print(
                "Source:",
                test_result.get(
                    "actual_source"
                )
            )

        print(
            "Reason:",
            test_result.get(
                "reason",
                ""
            )
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_tests = len(results)

    passed = sum(
        1
        for result in results
        if result.get("passed")
    )

    failed = sum(
        1
        for result in results
        if (
            not result.get("passed")
            and
            result.get("status")
            != "SKIPPED"
        )
    )

    skipped = sum(
        1
        for result in results
        if result.get("status") == "SKIPPED"
    )

    executed = total_tests - skipped

    accuracy = (
        passed / executed * 100
        if executed > 0
        else 0.0
    )

    summary = {
        "total_tests": total_tests,
        "executed_tests": executed,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "accuracy_percent": round(
            accuracy,
            2
        ),
    }

    # ========================================================
    # FINAL OBJECT
    # ========================================================

    output = {

        "project": "Pulmo Guide",

        "day": 4,

        "component": "Confidence Gate",

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "configuration": {

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "top_score_weight":
                TOP_SCORE_WEIGHT,

            "average_score_weight":
                AVERAGE_SCORE_WEIGHT,

            "evidence_level_weight":
                EVIDENCE_LEVEL_WEIGHT,

            "patient_pdf": str(
                PATIENT_PDF
            ),

            "patient_pdf_exists":
                PATIENT_PDF.exists(),
        },

        "summary": summary,

        "tests": results,
    }

    return output


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    output: Dict[str, Any]
):

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    output: Dict[str, Any]
):

    summary = output["summary"]

    lines = []

    lines.append(
        "PULMO GUIDE — DAY 4"
    )

    lines.append(
        "CONFIDENCE GATE REPORT"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        f"Generated at: "
        f"{output['generated_at']}"
    )

    lines.append(
        f"Confidence threshold: "
        f"{output['configuration']['confidence_threshold']}"
    )

    lines.append("")

    lines.append(
        "SUMMARY"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Total tests: "
        f"{summary['total_tests']}"
    )

    lines.append(
        f"Executed: "
        f"{summary['executed_tests']}"
    )

    lines.append(
        f"Passed: "
        f"{summary['passed']}"
    )

    lines.append(
        f"Failed: "
        f"{summary['failed']}"
    )

    lines.append(
        f"Skipped: "
        f"{summary['skipped']}"
    )

    lines.append(
        f"Accuracy: "
        f"{summary['accuracy_percent']:.2f}%"
    )

    lines.append("")

    lines.append(
        "TEST DETAILS"
    )

    lines.append(
        "-" * 70
    )

    for test in output["tests"]:

        lines.append("")

        status = (
            "PASS"
            if test.get("passed")
            else
            (
                "SKIPPED"
                if test.get("status")
                == "SKIPPED"
                else
                "FAIL"
            )
        )

        lines.append(
            f"{test['test_id']} — "
            f"{test['name']}"
        )

        lines.append(
            f"Status: {status}"
        )

        lines.append(
            f"Query: {test['query']}"
        )

        if test.get(
            "actual_source"
        ):

            lines.append(
                f"Source: "
                f"{test['actual_source']}"
            )

        if "confidence" in test:

            lines.append(
                f"Confidence: "
                f"{test['confidence']}"
            )

        if "top_score" in test:

            lines.append(
                f"Top score: "
                f"{test['top_score']}"
            )

        if "average_score" in test:

            lines.append(
                f"Average score: "
                f"{test['average_score']}"
            )

        lines.append(
            f"Reason: "
            f"{test.get('reason', '')}"
        )

    lines.append("")
    lines.append("=" * 70)

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        output = run_all_tests()

        save_json(
            output
        )

        save_report(
            output
        )

        summary = output["summary"]

        print("\n" + "=" * 70)

        print(
            f"\nTotal tests: "
            f"{summary['total_tests']}"
        )

        print(
            f"Passed: "
            f"{summary['passed']}"
        )

        print(
            f"Failed: "
            f"{summary['failed']}"
        )

        print(
            f"Skipped: "
            f"{summary['skipped']}"
        )

        print(
            f"Accuracy: "
            f"{summary['accuracy_percent']:.2f}%"
        )

        print("\nJSON saved to:")
        print(
            JSON_OUTPUT
        )

        print("\nReport saved to:")
        print(
            REPORT_OUTPUT
        )

        print(
            "\nConfidence Gate tests completed."
        )

    except Exception as error:

        print(
            "\nCONFIDENCE GATE FAILED."
        )

        print(
            type(error).__name__,
            ":",
            str(error)
        )

        traceback.print_exc()

        sys.exit(1)
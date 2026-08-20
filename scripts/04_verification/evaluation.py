"""
============================================================
PULMO GUIDE
DAY 4 — COMPREHENSIVE FINAL EVALUATION (CORRECTED)
============================================================
Deterministic. No LLM. No LangChain. Safe with missing files.
Flexible JSON parsing tuned to the ACTUAL schemas found in
this project's evaluation artifacts.
============================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
FINAL_RESULTS_FILE = EVALUATION_DIR / "final_evaluation_results.json"
FINAL_REPORT_FILE = EVALUATION_DIR / "final_evaluation_report.txt"


# ============================================================
# FILE REGISTRY
# ============================================================

FILES = {
    "claim_verification": EVALUATION_DIR / "claim_verification_results.json",
    "claim_extraction": EVALUATION_DIR / "claim_extraction_results.json",
    "citation_check": EVALUATION_DIR / "citation_check_results.json",
    "faithfulness": EVALUATION_DIR / "faithfulness_results.json",
    "confidence_gate": EVALUATION_DIR / "confidence_gate_results.json",
    "retrieval": EVALUATION_DIR / "retrieval_evaluation_results.json",
    "hybrid_retrieval": EVALUATION_DIR / "hybrid_retrieval_evaluation_results.json",
    "day3_safety": EVALUATION_DIR / "day3_safety_results.json",
    "day3_test": EVALUATION_DIR / "day3_test_results.json",
    "refusal_threshold": EVALUATION_DIR / "refusal_threshold_results.json",
    "refusal_evaluation": EVALUATION_DIR / "refusal_evaluation_set.json",
    "threshold_experiment": EVALUATION_DIR / "threshold_experiment_results.json",
    "latency": EVALUATION_DIR / "latency_results.json",
    "evaluation_set": EVALUATION_DIR / "evaluation_set.json",
}


# ============================================================
# GENERAL UTILITIES
# ============================================================

def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"WARNING: Could not load {path.name}: {exc}")
        return None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def percentage(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return round((part / total) * 100, 2)


def normalize_metric(value: Optional[float]) -> Optional[float]:
    """0-1 values become percentages. Already-percent values stay."""
    if value is None:
        return None
    if 0 <= value <= 1:
        return round(value * 100, 2)
    return round(value, 2)


def first_existing(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def get_summary(data: Optional[Any]) -> Dict[str, Any]:
    """Extract summary object. Handles 'summary' key only.
    Files that store totals at top level are handled separately
    via get_top_level_or_summary()."""
    if isinstance(data, dict):
        summary = data.get("summary")
        if isinstance(summary, dict):
            return summary
    return {}


def get_top_level_or_summary(
    data: Optional[Any],
    summary: Dict[str, Any],
    keys: List[str],
    default: Any = None,
) -> Any:
    """
    Flexible lookup: try summary first, then the top level of the
    raw JSON. This is the key fix for files (like day3_safety) that
    store their totals directly at the top level with NO 'summary'
    wrapper.
    """
    value = first_existing(summary, keys, None)
    if value is not None:
        return value
    if isinstance(data, dict):
        value = first_existing(data, keys, None)
        if value is not None:
            return value
    return default


def get_results_list(data: Optional[Any]) -> List[Dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in [
            "results", "test_results", "evaluations", "cases",
            "items", "data", "tests",
            # project-specific keys found in the actual JSON files:
            "scored_items", "per_question_results", "question_results",
        ]:
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def get_total_tests(data: Optional[Any]) -> int:
    results = get_results_list(data)
    summary = get_summary(data)

    value = get_top_level_or_summary(
        data, summary,
        ["total_tests", "total_cases", "total_questions", "count", "num_questions"],
        None,
    )
    if value is not None:
        return safe_int(value)

    return len(results)


def count_statuses(
    data: Optional[Any],
    positive_statuses: set,
    negative_statuses: set,
) -> Dict[str, int]:
    """Count pass/fail using a string status field."""
    results = get_results_list(data)
    passed = 0
    failed = 0
    for result in results:
        status = str(
            first_existing(result, ["result", "status", "overall"], "")
        ).upper().strip()
        if status in positive_statuses:
            passed += 1
        elif status in negative_statuses:
            failed += 1
    return {"passed": passed, "failed": failed}


def count_boolean_passed(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count pass/fail using a boolean 'passed' field per result item.
    Needed for day3_safety_results.json / day3_test_results.json where
    each item carries 'passed': true/false rather than a status string."""
    passed = 0
    failed = 0
    for result in results:
        val = result.get("passed") if isinstance(result, dict) else None
        if val is True:
            passed += 1
        elif val is False:
            failed += 1
    return {"passed": passed, "failed": failed}


def load_all_files() -> Dict[str, Any]:
    loaded = {}
    print("Loading evaluation files...")
    print("-" * 78)
    for name, path in FILES.items():
        data = load_json(path)
        loaded[name] = data
        if data is None:
            print(f"[MISSING] {name:<24} {path.name}")
        else:
            print(f"[OK]      {name:<24} {path.name}")
    print()
    return loaded


# ============================================================
# DATASET DEPTH
# ============================================================

def evaluate_dataset_depth(data: Optional[Any]) -> Dict[str, Any]:
    count = get_total_tests(data)
    minimum = 20
    return {
        "available": data is not None,
        "test_cases": count,
        "recommended_minimum": minimum,
        "meets_20_question_target": count >= minimum,
        "completion_percentage": min(percentage(count, minimum), 100.0),
    }


# ============================================================
# CLAIM VERIFICATION  (schema already matches code -- kept, hardened)
# ============================================================

def evaluate_claim_verification(data: Optional[Any]) -> Dict[str, Any]:
    if data is None:
        return {
            "available": False, "total_tests": 0, "passed_tests": 0,
            "test_accuracy": None, "total_claims": 0, "supported_claims": 0,
            "partially_supported_claims": 0, "unsupported_claims": 0,
            "claim_support_rate": None, "claim_coverage_rate": None,
        }

    results = get_results_list(data)
    summary = get_summary(data)
    total_tests = get_total_tests(data)

    passed_tests = safe_int(
        get_top_level_or_summary(data, summary, ["passed_tests", "passed"], 0)
    )

    if passed_tests == 0 and results:
        counts = count_statuses(
            data,
            {"SUPPORTED", "PASS", "PASSED", "NO_CLAIMS"},
            {"UNSUPPORTED", "FAIL", "FAILED", "PARTIALLY_SUPPORTED"},
        )
        passed_tests = counts["passed"]

    total_claims = safe_int(
        get_top_level_or_summary(data, summary, ["total_claims", "claims"], 0)
    )
    supported_claims = safe_int(
        get_top_level_or_summary(data, summary, ["supported_claims", "supported"], 0)
    )
    partially_supported = safe_int(
        get_top_level_or_summary(
            data, summary, ["partially_supported_claims", "partially_supported"], 0
        )
    )
    unsupported_claims = safe_int(
        get_top_level_or_summary(data, summary, ["unsupported_claims", "unsupported"], 0)
    )

    if total_claims == 0:
        for result in results:
            claims = first_existing(result, ["claims", "claim_results"], [])
            if not isinstance(claims, list):
                continue
            total_claims += len(claims)
            for claim in claims:
                status = str(
                    first_existing(claim, ["status", "verification", "result"], "")
                ).upper().strip()
                if status == "SUPPORTED":
                    supported_claims += 1
                elif status == "PARTIALLY_SUPPORTED":
                    partially_supported += 1
                elif status == "UNSUPPORTED":
                    unsupported_claims += 1

    return {
        "available": True,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "test_accuracy": percentage(passed_tests, total_tests),
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "partially_supported_claims": partially_supported,
        "unsupported_claims": unsupported_claims,
        "claim_support_rate": (
            percentage(supported_claims, total_claims) if total_claims > 0 else None
        ),
        "claim_coverage_rate": (
            percentage(supported_claims + partially_supported, total_claims)
            if total_claims > 0 else None
        ),
    }


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def evaluate_claim_extraction(data: Optional[Any]) -> Dict[str, Any]:
    if data is None:
        return {"available": False, "test_cases": 0, "claims_extracted": 0}

    results = get_results_list(data)
    summary = get_summary(data)

    claims = safe_int(
        get_top_level_or_summary(
            data, summary, ["total_claims", "claims_extracted", "claim_count"], 0
        )
    )

    if claims == 0:
        for result in results:
            claim_list = first_existing(result, ["claims", "claim_results"], [])
            if isinstance(claim_list, list):
                claims += len(claim_list)

    return {"available": True, "test_cases": len(results), "claims_extracted": claims}


# ============================================================
# CITATION CHECK
# ============================================================

def evaluate_citations(data: Optional[Any]) -> Dict[str, Any]:
    if data is None:
        return {"available": False, "total_tests": 0, "passed": 0, "failed": 0,
                "citation_accuracy": None}

    results = get_results_list(data)
    summary = get_summary(data)
    total_tests = get_total_tests(data)

    passed = safe_int(get_top_level_or_summary(data, summary, ["passed", "passed_tests"], 0))
    failed = safe_int(get_top_level_or_summary(data, summary, ["failed", "failed_tests"], 0))

    if passed == 0 and failed == 0 and results:
        counts = count_statuses(
            data,
            {"PASS", "PASSED", "SUPPORTED", "VALID", "CORRECT"},
            {"FAIL", "FAILED", "INVALID", "UNSUPPORTED", "INCORRECT"},
        )
        passed, failed = counts["passed"], counts["failed"]

    if passed == 0 and failed == 0 and results:
        counts = count_boolean_passed(results)
        passed, failed = counts["passed"], counts["failed"]

    return {
        "available": True,
        "total_tests": total_tests,
        "passed": passed,
        "failed": failed,
        "citation_accuracy": percentage(passed, total_tests),
    }


# ============================================================
# FAITHFULNESS
# ============================================================

def evaluate_faithfulness(data: Optional[Any]) -> Dict[str, Any]:
    if data is None:
        return {"available": False, "total_tests": 0, "faithful": 0,
                "partially_faithful": 0, "unfaithful": 0,
                "strict_rate": None, "broad_rate": None}

    results = get_results_list(data)
    summary = get_summary(data)
    total_tests = get_total_tests(data)

    faithful = safe_int(get_top_level_or_summary(data, summary, ["faithful"], 0))
    partially = safe_int(get_top_level_or_summary(data, summary, ["partially_faithful"], 0))
    unfaithful = safe_int(get_top_level_or_summary(data, summary, ["unfaithful"], 0))

    if faithful == 0 and partially == 0 and unfaithful == 0 and results:
        for result in results:
            status = str(
                first_existing(result, ["result", "status"], "")
            ).upper().strip()
            if status == "FAITHFUL":
                faithful += 1
            elif status == "PARTIALLY_FAITHFUL":
                partially += 1
            elif status == "UNFAITHFUL":
                unfaithful += 1

    return {
        "available": True,
        "total_tests": total_tests,
        "faithful": faithful,
        "partially_faithful": partially,
        "unfaithful": unfaithful,
        "strict_rate": percentage(faithful, total_tests),
        "broad_rate": percentage(faithful + partially, total_tests),
    }


# ============================================================
# CONFIDENCE GATE
# ============================================================

def evaluate_confidence_gate(data: Optional[Any]) -> Dict[str, Any]:
    if data is None:
        return {"available": False, "total_tests": 0, "passed": 0, "failed": 0,
                "accuracy": None}

    results = get_results_list(data)
    summary = get_summary(data)
    total_tests = get_total_tests(data)

    passed = safe_int(
        get_top_level_or_summary(data, summary, ["passed", "passed_tests"], 0)
    )
    failed = safe_int(
        get_top_level_or_summary(data, summary, ["failed", "failed_tests"], 0)
    )

    if passed == 0 and failed == 0 and results:
        counts = count_statuses(
            data,
            {"PASS", "PASSED", "ACCEPT", "ACCEPTED", "CORRECT"},
            {"FAIL", "FAILED", "REJECT", "REJECTED", "INCORRECT"},
        )
        passed, failed = counts["passed"], counts["failed"]

    if passed == 0 and failed == 0 and results:
        # confidence_gate_results.json stores a boolean "passed" per test,
        # and also uses "gate": "PASS"/"FAIL" -- try both.
        counts = count_boolean_passed(results)
        passed, failed = counts["passed"], counts["failed"]
        if passed == 0 and failed == 0:
            counts = count_statuses(
                {"results": [{"status": r.get("gate")} for r in results]},
                {"PASS"}, {"FAIL"},
            )
            passed, failed = counts["passed"], counts["failed"]

    return {
        "available": True,
        "total_tests": total_tests,
        "passed": passed,
        "failed": failed,
        "accuracy": percentage(passed, total_tests),
    }


# ============================================================
# RETRIEVAL EVALUATION
# retrieval_evaluation_results.json actual schema:
#   config: {..., num_questions}
#   averages: { semantic:{...}, bm25:{...}, hybrid:{...}, hybrid_rerank:{...} }
#   per_question_results: [ {question, ground_truth_ids, configs:{...}} ]
# Real metric field names: precision_at_5, recall_at_5, hit_at_5, mrr_at_5
# (no ndcg / map available in the data -> left as None, never invented)
# ============================================================

RETRIEVAL_METRIC_KEYS = {
    "precision_at_5": "precision",
    "precision_at_10": "precision_at_10",
    "recall_at_5": "recall",
    "recall_at_10": "recall_at_10",
    "hit_at_5": "hit_rate",
    "mrr_at_5": "mrr",
    "ndcg_at_5": "ndcg",
    "map_at_5": "map",
}


def extract_retrieval_metric_dict(values: Any) -> Dict[str, float]:
    metrics = {}
    if not isinstance(values, dict):
        return metrics
    for src_key, out_key in RETRIEVAL_METRIC_KEYS.items():
        val = safe_float(values.get(src_key))
        if val is not None:
            metrics[out_key] = normalize_metric(val)
    return metrics


def evaluate_retrieval(data: Optional[Any]) -> Dict[str, Any]:
    if data is None:
        return {
            "available": False, "test_cases": 0, "configs": {},
            "best_config": None, "retrieval_score": None,
        }

    config = data.get("config", {}) if isinstance(data, dict) else {}
    averages = data.get("averages", {}) if isinstance(data, dict) else {}
    per_question = data.get("per_question_results", []) if isinstance(data, dict) else []

    test_cases = safe_int(config.get("num_questions")) or len(per_question)

    configs: Dict[str, Any] = {}

    if isinstance(averages, dict) and averages and all(
        isinstance(v, dict) for v in averages.values()
    ):
        # multi-config format: semantic / bm25 / hybrid / hybrid_rerank
        for name, vals in averages.items():
            metrics = extract_retrieval_metric_dict(vals)
            if metrics:
                score = round(sum(metrics.values()) / len(metrics), 2)
                configs[name] = {**metrics, "score": score}
    elif isinstance(averages, dict) and averages:
        # flat single-config format (used by hybrid_retrieval file)
        metrics = extract_retrieval_metric_dict(averages)
        if metrics:
            score = round(sum(metrics.values()) / len(metrics), 2)
            configs["default"] = {**metrics, "score": score}

    best_name = None
    best_score = None
    for name, vals in configs.items():
        score = vals.get("score")
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_name = name

    return {
        "available": bool(configs),
        "test_cases": test_cases,
        "configs": configs,
        "best_config": best_name,
        "retrieval_score": best_score,
    }


def evaluate_hybrid_retrieval(data: Optional[Any]) -> Dict[str, Any]:
    """
    hybrid_retrieval_evaluation_results.json is a standalone, detailed
    trace of the Hybrid 70/30 run (same metrics as the 'hybrid' entry
    inside retrieval_evaluation_results.json, plus full per-chunk text).
    Used here as a cross-validation / detailed-evidence source, NOT
    averaged into the primary retrieval score (to avoid double counting
    the same run twice).
    """
    if data is None:
        return {"available": False, "test_cases": 0, "metrics": {}, "score": None}

    config = data.get("config", {}) if isinstance(data, dict) else {}
    averages = data.get("averages", {}) if isinstance(data, dict) else {}
    per_question = data.get("per_question_results", []) if isinstance(data, dict) else []

    test_cases = safe_int(config.get("num_questions")) or len(per_question)
    metrics = extract_retrieval_metric_dict(averages)
    score = round(sum(metrics.values()) / len(metrics), 2) if metrics else None

    return {
        "available": bool(metrics),
        "test_cases": test_cases,
        "metrics": metrics,
        "score": score,
        "reranker_used": bool(config.get("reranker")),
    }


def calculate_retrieval_quality(retrieval: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieval quality = the best-performing configuration found in
    retrieval_evaluation_results.json (semantic / bm25 / hybrid /
    hybrid_rerank), selected purely by highest average of its own
    available metrics. No invented metrics, no blind averaging across
    configs that would double count the same questions.
    """
    if not retrieval.get("available") or retrieval.get("retrieval_score") is None:
        return {
            "available": False,
            "score": None,
            "selected_config": None,
            "configs": retrieval.get("configs", {}),
        }

    return {
        "available": True,
        "score": retrieval["retrieval_score"],
        "selected_config": retrieval["best_config"],
        "configs": retrieval.get("configs", {}),
    }


# ============================================================
# SAFETY  (day3_safety_results.json — top-level totals, boolean 'passed')
# ============================================================

def evaluate_safety(data: Optional[Any]) -> Dict[str, Any]:
    if data is None:
        return {"available": False, "test_cases": 0, "passed": 0, "failed": 0,
                "accuracy": None}

    summary = get_summary(data)
    total = get_total_tests(data)
    results = get_results_list(data)

    passed = safe_int(
        get_top_level_or_summary(data, summary, ["passed_tests", "passed"], 0)
    )
    failed = safe_int(
        get_top_level_or_summary(data, summary, ["failed_tests", "failed"], 0)
    )

    if passed == 0 and failed == 0 and results:
        counts = count_statuses(
            data,
            {"PASS", "PASSED", "SAFE", "CORRECT", "ACCEPTED"},
            {"FAIL", "FAILED", "UNSAFE", "INCORRECT", "REJECTED"},
        )
        passed, failed = counts["passed"], counts["failed"]

    if passed == 0 and failed == 0 and results:
        counts = count_boolean_passed(results)
        passed, failed = counts["passed"], counts["failed"]

    return {
        "available": True,
        "test_cases": total,
        "passed": passed,
        "failed": failed,
        "accuracy": percentage(passed, total),
    }


def evaluate_day3_test_suite(data: Optional[Any]) -> Dict[str, Any]:
    """
    day3_test_results.json is a SEPARATE Day-3 suite (grounded
    generation / citation / refusal / adversarial). It overlaps
    conceptually with claim/citation/faithfulness/refusal evaluations,
    so it is reported as supplementary evidence only and is NOT folded
    into any weighted score (avoids double counting the same behavior).
    """
    if data is None:
        return {"available": False, "total_tests": 0, "passed": 0, "failed": 0,
                "pass_rate": None}

    total = safe_int(first_existing(data, ["total_tests"], 0))
    passed = safe_int(first_existing(data, ["passed"], 0))
    failed = safe_int(first_existing(data, ["failed"], 0))
    pass_rate = safe_float(first_existing(data, ["pass_rate"], None))

    return {
        "available": True,
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (
            normalize_metric(pass_rate) if pass_rate is not None
            else percentage(passed, total)
        ),
    }


# ============================================================
# REFUSAL
# refusal_threshold_results.json actual schema:
#   scored_items: [ {id, type, question, expected_decision, hybrid_score, ...} ]
#   threshold_results: [ {threshold, accepted, rejected, correct_accept,
#                          correct_refuse, false_accept, false_reject,
#                          precision, recall, accuracy} ]
# There is no single 'accuracy' field for the file -- the correct
# reading is: pick the best-performing threshold from threshold_results.
# ============================================================

def evaluate_refusal_threshold(data: Optional[Any]) -> Dict[str, Any]:
    if data is None:
        return {"available": False, "test_cases": 0, "selected_threshold": None,
                "accuracy": None, "precision": None, "recall": None,
                "thresholds_tested": 0}

    scored_items = data.get("scored_items", []) if isinstance(data, dict) else []
    threshold_results = data.get("threshold_results", []) if isinstance(data, dict) else []

    test_cases = len(scored_items)

    if not threshold_results:
        return {"available": False, "test_cases": test_cases, "selected_threshold": None,
                "accuracy": None, "precision": None, "recall": None,
                "thresholds_tested": 0}

    best = max(
        threshold_results,
        key=lambda t: safe_float(t.get("accuracy")) if safe_float(t.get("accuracy")) is not None else -1.0,
    )

    return {
        "available": True,
        "test_cases": test_cases,
        "selected_threshold": best.get("threshold"),
        "accuracy": normalize_metric(safe_float(best.get("accuracy"))),
        "precision": normalize_metric(safe_float(best.get("precision"))),
        "recall": normalize_metric(safe_float(best.get("recall"))),
        "correct_accept": best.get("correct_accept"),
        "correct_refuse": best.get("correct_refuse"),
        "false_accept": best.get("false_accept"),
        "false_reject": best.get("false_reject"),
        "thresholds_tested": len(threshold_results),
    }


def evaluate_refusal_ground_truth_only(data: Optional[Any]) -> Dict[str, Any]:
    """
    refusal_evaluation_set.json is only ground truth (question +
    expected_decision), no actual system outcome -- it CANNOT produce
    an accuracy number. Used only to report test-case count when the
    real results file (refusal_threshold_results.json) is missing.
    """
    if data is None:
        return {"available": False, "test_cases": 0}
    results = get_results_list(data)
    return {"available": False, "test_cases": len(results)}


def choose_refusal_metrics(loaded: Dict[str, Any]) -> Dict[str, Any]:
    threshold_data = loaded.get("refusal_threshold")
    if threshold_data is not None:
        metrics = evaluate_refusal_threshold(threshold_data)
        if metrics.get("available"):
            return metrics

    # fallback: ground-truth-only set, no accuracy obtainable
    return evaluate_refusal_ground_truth_only(loaded.get("refusal_evaluation"))


# ============================================================
# THRESHOLD EXPERIMENT (confidence-gate acceptance, in-KB only)
# threshold_experiment_results.json schema:
#   threshold_results: [ {threshold, accepted, rejected, acceptance_rate,
#                          correct_accepts, false_accepts, missed_relevant,
#                          accept_precision, relevant_recall} ]
#   question_results: [ {question, ground_truth_ids, top_score,
#                         has_relevant_chunk, results:[...]} ]
# This is a SEPARATE experiment from refusal (in-KB acceptance quality,
# not accept/refuse of out-of-KB questions). Reported standalone.
# ============================================================

def evaluate_threshold_experiment(data: Optional[Any]) -> Dict[str, Any]:
    if data is None:
        return {"available": False}

    threshold_results = data.get("threshold_results", []) if isinstance(data, dict) else []
    question_results = data.get("question_results", []) if isinstance(data, dict) else []

    if not threshold_results:
        return {"available": False}

    best = max(
        threshold_results,
        key=lambda t: safe_float(t.get("accept_precision"))
        if safe_float(t.get("accept_precision")) is not None else -1.0,
    )

    distinct_outcomes = {
        (t.get("accepted"), t.get("rejected")) for t in threshold_results
    }

    return {
        "available": True,
        "test_cases": len(question_results),
        "selected_threshold": best.get("threshold"),
        "accept_precision": normalize_metric(safe_float(best.get("accept_precision"))),
        "relevant_recall": normalize_metric(safe_float(best.get("relevant_recall"))),
        "acceptance_rate": normalize_metric(safe_float(best.get("acceptance_rate"))),
        "thresholds_tested": len(threshold_results),
        "all_thresholds_identical": len(distinct_outcomes) == 1,
    }


# ============================================================
# LATENCY
# ============================================================

def evaluate_latency(data: Optional[Any]) -> Dict[str, Any]:
    if data is None:
        return {"available": False, "test_cases": 0, "average_ms": None,
                "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}

    results = get_results_list(data)
    summary = get_summary(data)
    stats = data.get("statistics", {}) if isinstance(data, dict) else {}

    def find(names: List[str]) -> Optional[float]:
        value = get_top_level_or_summary(data, summary, names, None)
        if value is None and isinstance(stats, dict):
            value = first_existing(stats, names, None)
        return safe_float(value)

    average = find(["average_ms", "avg_ms", "mean_ms", "average_latency_ms"])
    median = find(["median_ms", "p50_ms", "median_latency_ms"])
    p95 = find(["p95_ms", "p95_latency_ms"])
    minimum = find(["min_ms", "minimum_ms"])
    maximum = find(["max_ms", "maximum_ms"])

    values = []
    for result in results:
        value = first_existing(
            result, ["latency_ms", "response_time_ms", "duration_ms"], None
        )
        value = safe_float(value)
        if value is not None:
            values.append(value)

    if values:
        values_sorted = sorted(values)
        if average is None:
            average = sum(values) / len(values)
        if median is None:
            middle = len(values_sorted) // 2
            if len(values_sorted) % 2 == 1:
                median = values_sorted[middle]
            else:
                median = (values_sorted[middle - 1] + values_sorted[middle]) / 2
        if minimum is None:
            minimum = min(values)
        if maximum is None:
            maximum = max(values)
        if p95 is None:
            rank = max(1, int(0.95 * len(values_sorted) + 0.999999))
            rank = min(rank, len(values_sorted))
            p95 = values_sorted[rank - 1]

    test_cases = safe_int(stats.get("count")) if isinstance(stats, dict) else 0
    if not test_cases:
        test_cases = len(results)

    return {
        "available": True,
        "test_cases": test_cases,
        "average_ms": round(average, 2) if average is not None else None,
        "median_ms": round(median, 2) if median is not None else None,
        "p95_ms": round(p95, 2) if p95 is not None else None,
        "min_ms": round(minimum, 2) if minimum is not None else None,
        "max_ms": round(maximum, 2) if maximum is not None else None,
    }


# ============================================================
# GROUNDING SCORE  (unaffected by the bugs -- logic kept identical)
# ============================================================

def calculate_grounding_score(
    claim_metrics: Dict[str, Any],
    citation_metrics: Dict[str, Any],
    faithfulness_metrics: Dict[str, Any],
    confidence_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    components = []

    claim_value = claim_metrics.get("claim_support_rate")
    if claim_metrics.get("available") and claim_value is not None:
        components.append((claim_value, 0.35))

    citation_value = citation_metrics.get("citation_accuracy")
    if citation_metrics.get("available") and citation_value is not None:
        components.append((citation_value, 0.30))

    faithfulness_value = faithfulness_metrics.get("strict_rate")
    if faithfulness_metrics.get("available") and faithfulness_value is not None:
        components.append((faithfulness_value, 0.25))

    confidence_value = confidence_metrics.get("accuracy")
    if confidence_metrics.get("available") and confidence_value is not None:
        components.append((confidence_value, 0.10))

    if not components:
        return {"available": False, "score": None, "percentage": None, "components": {}}

    total_weight = sum(weight for _, weight in components)
    weighted_score = sum(value * weight for value, weight in components) / total_weight

    return {
        "available": True,
        "score": round(weighted_score / 100, 4),
        "percentage": round(weighted_score, 2),
        "components": {
            "claim_support": (
                {"value": claim_value, "weight": 0.35}
                if claim_metrics.get("available") and claim_value is not None else None
            ),
            "citation_accuracy": (
                {"value": citation_value, "weight": 0.30}
                if citation_metrics.get("available") and citation_value is not None else None
            ),
            "strict_faithfulness": (
                {"value": faithfulness_value, "weight": 0.25}
                if faithfulness_metrics.get("available") and faithfulness_value is not None else None
            ),
            "confidence_gate": (
                {"value": confidence_value, "weight": 0.10}
                if confidence_metrics.get("available") and confidence_value is not None else None
            ),
        },
    }


# ============================================================
# SAFETY / UX SCORE
# ============================================================

def calculate_safety_score(safety: Dict[str, Any], refusal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Confidence gate is intentionally excluded (already counted in
    grounding). Threshold experiment and day3_test are reported as
    supplementary evidence only, not averaged in here, to avoid
    double-counting overlapping behavior.
    """
    components = []

    safety_value = safety.get("accuracy")
    if safety.get("available") and safety_value is not None:
        components.append(safety_value)

    refusal_value = refusal.get("accuracy")
    if refusal.get("available") and refusal_value is not None:
        components.append(refusal_value)

    if not components:
        return {"available": False, "score": None}

    return {"available": True, "score": round(sum(components) / len(components), 2)}


# ============================================================
# MEASURABLE PROJECT SCORE
# ============================================================

def calculate_measurable_score(
    retrieval_quality: Dict[str, Any],
    grounding_score: Dict[str, Any],
    safety_score: Dict[str, Any],
    dataset_depth: Dict[str, Any],
) -> Dict[str, Any]:
    components = []

    retrieval_value = retrieval_quality.get("score")
    if retrieval_quality.get("available") and retrieval_value is not None:
        components.append((retrieval_value, 30))

    grounding_value = grounding_score.get("percentage")
    if grounding_score.get("available") and grounding_value is not None:
        components.append((grounding_value, 25))

    evaluation_value = dataset_depth.get("completion_percentage")
    if dataset_depth.get("available"):
        components.append((evaluation_value, 15))

    safety_value = safety_score.get("score")
    if safety_score.get("available") and safety_value is not None:
        components.append((safety_value, 15))

    if not components:
        return {
            "available": False, "score": None, "max_measurable_weight": 85,
            "note": "No measurable evaluation evidence.",
        }

    weighted_sum = sum(value * weight for value, weight in components)
    total_weight = sum(weight for _, weight in components)

    return {
        "available": True,
        "score": round(weighted_sum / total_weight, 2),
        "weighted_points": round(weighted_sum / 100, 2),
        "max_available_weight": total_weight,
        "official_architecture_weight": 15,
        "note": (
            "Project-level measurable indicator only; "
            "not the official hackathon score."
        ),
    }


# ============================================================
# JUDGE-ORIENTED RUBRIC
# ============================================================

def build_rubric_summary(
    retrieval_quality: Dict[str, Any],
    grounding_score: Dict[str, Any],
    safety_score: Dict[str, Any],
    dataset_depth: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "retrieval_quality_30_percent": {
            "weight": 30,
            "measured_score": retrieval_quality.get("score"),
            "status": "MEASURED" if retrieval_quality.get("available") else "NOT_AVAILABLE",
            "selected_configuration": retrieval_quality.get("selected_config"),
            "components": [
                "Semantic retrieval", "BM25 retrieval", "Hybrid 70/30",
                "Hybrid + Reranking", "Precision@5", "Recall@5",
                "Hit Rate@5", "MRR@5",
            ],
        },
        "grounding_citation_25_percent": {
            "weight": 25,
            "measured_score": grounding_score.get("percentage"),
            "status": "MEASURED" if grounding_score.get("available") else "NOT_AVAILABLE",
            "components": ["Claim support", "Citation accuracy", "Faithfulness", "Confidence gate"],
        },
        "system_architecture_15_percent": {
            "weight": 15,
            "measured_score": None,
            "status": "JUDGE_REVIEW_REQUIRED",
            "note": (
                "Requires human inspection of architecture, modularity, "
                "separation of components, diagrams, and code organization."
            ),
        },
        "evaluation_depth_15_percent": {
            "weight": 15,
            "measured_score": dataset_depth.get("completion_percentage"),
            "status": (
                "MEETS_TARGET" if dataset_depth.get("meets_20_question_target")
                else "BELOW_20_CASES"
            ),
            "test_cases": dataset_depth.get("test_cases", 0),
            "recommended_minimum": 20,
        },
        "safety_ux_15_percent": {
            "weight": 15,
            "measured_score": safety_score.get("score"),
            "status": "MEASURED" if safety_score.get("available") else "NOT_AVAILABLE",
            "components": [
                "Safety tests", "Refusal evaluation (best threshold)",
                "Confidence gate evidence (see grounding)",
                "Threshold experiment (supplementary)", "Latency evidence",
            ],
        },
        "judge_note": (
            "These are quantitative evidence summaries and not official "
            "hackathon judge scores. Architecture, live UX, clinical "
            "disclaimer, citation display, and demo quality require "
            "human inspection."
        ),
    }


# ============================================================
# FAILURE ANALYSIS
# ============================================================

def analyze_claim_failures(data: Optional[Any]) -> List[Dict[str, Any]]:
    failures = []
    if data is None:
        return failures
    results = get_results_list(data)

    for result in results:
        test_id = first_existing(result, ["id", "test_id"], "UNKNOWN")
        question = result.get("question", "")
        overall = str(
            first_existing(result, ["overall", "result", "status"], "")
        ).upper().strip()

        if overall in {"UNSUPPORTED", "PARTIALLY_SUPPORTED"}:
            failures.append({
                "type": "claim_verification", "id": test_id,
                "question": question, "status": overall,
            })

        claims = first_existing(result, ["claims", "claim_results"], [])
        if not isinstance(claims, list):
            continue

        for claim in claims:
            status = str(
                first_existing(claim, ["status", "verification", "result"], "")
            ).upper().strip()
            if status in {"UNSUPPORTED", "PARTIALLY_SUPPORTED"}:
                failures.append({
                    "type": "claim",
                    "id": first_existing(claim, ["id", "claim_id"], test_id),
                    "question": question,
                    "status": status,
                    "evidence": first_existing(claim, ["evidence", "evidence_id"], "N/A"),
                })

    return failures


def analyze_faithfulness_failures(data: Optional[Any]) -> List[Dict[str, Any]]:
    failures = []
    if data is None:
        return failures

    for result in get_results_list(data):
        status = str(
            first_existing(result, ["result", "status"], "")
        ).upper().strip()
        if status not in {"PARTIALLY_FAITHFUL", "UNFAITHFUL"}:
            continue
        failures.append({
            "type": "faithfulness",
            "id": first_existing(result, ["id", "test_id"], "UNKNOWN"),
            "question": result.get("question", ""),
            "status": status,
            "score": safe_float(result.get("score", 0)),
            "evidence": result.get("evidence", "N/A"),
        })
    return failures


def analyze_retrieval_failures(data: Optional[Any]) -> List[Dict[str, Any]]:
    """
    Uses per_question_results with the 'hybrid' config's hit_at_5
    (miss = no relevant chunk retrieved in top 5) since that is the
    selected/final configuration.
    """
    failures = []
    if data is None:
        return failures

    per_question = data.get("per_question_results", []) if isinstance(data, dict) else []

    for idx, result in enumerate(per_question):
        configs = result.get("configs", {})
        hybrid = configs.get("hybrid", {}) if isinstance(configs, dict) else {}
        hit = hybrid.get("hit_at_5")
        if hit is not None and safe_int(hit) == 0:
            failures.append({
                "type": "retrieval",
                "id": f"Q{idx + 1}",
                "question": result.get("question", ""),
                "status": "MISS_HYBRID_CONFIG",
            })

    return failures


def analyze_generic_failures(
    data: Optional[Any], failure_statuses: set, failure_type: str,
) -> List[Dict[str, Any]]:
    failures = []
    if data is None:
        return failures

    for result in get_results_list(data):
        status = str(
            first_existing(result, ["result", "status"], "")
        ).upper().strip()
        if status not in failure_statuses:
            continue
        failures.append({
            "type": failure_type,
            "id": first_existing(result, ["id", "test_id"], "UNKNOWN"),
            "question": result.get("question", ""),
            "status": status,
        })
    return failures


# ============================================================
# REPORT HELPERS
# ============================================================

def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


# ============================================================
# REPORT
# ============================================================

def build_report(
    loaded: Dict[str, Any],
    claim_metrics: Dict[str, Any],
    citation_metrics: Dict[str, Any],
    faithfulness_metrics: Dict[str, Any],
    confidence_metrics: Dict[str, Any],
    retrieval_metrics: Dict[str, Any],
    hybrid_metrics: Dict[str, Any],
    safety_metrics: Dict[str, Any],
    refusal_metrics: Dict[str, Any],
    threshold_experiment_metrics: Dict[str, Any],
    day3_test_metrics: Dict[str, Any],
    latency_metrics: Dict[str, Any],
    claim_extraction_metrics: Dict[str, Any],
    grounding_score: Dict[str, Any],
    retrieval_quality: Dict[str, Any],
    safety_score: Dict[str, Any],
    measurable_score: Dict[str, Any],
    dataset_depth: Dict[str, Any],
    rubric_summary: Dict[str, Any],
    claim_failures: List[Dict[str, Any]],
    faithfulness_failures: List[Dict[str, Any]],
    retrieval_failures: List[Dict[str, Any]],
    citation_failures: List[Dict[str, Any]],
    safety_failures: List[Dict[str, Any]],
) -> str:

    lines = []
    lines.append("=" * 78)
    lines.append("PULMO GUIDE — COMPREHENSIVE FINAL EVALUATION")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Deterministic evaluation — No LLM required")
    lines.append("")

    # -------------------- EXECUTIVE SUMMARY --------------------
    lines.append("1. EXECUTIVE SUMMARY")
    lines.append("-" * 78)
    lines.append("Overall Grounding Score: " + fmt(grounding_score.get("percentage")) + "%")
    lines.append("Retrieval Quality Score: " + fmt(retrieval_quality.get("score")) + "%")
    lines.append("Safety / Refusal Score: " + fmt(safety_score.get("score")) + "%")
    lines.append("Measurable Project Indicator: " + fmt(measurable_score.get("score")) + "%")
    lines.append("Evaluation Cases: " + str(dataset_depth.get("test_cases", 0)))
    lines.append(
        "20+ Dataset Target: "
        + ("MET" if dataset_depth.get("meets_20_question_target") else "NOT MET")
    )
    lines.append("")

    # -------------------- RETRIEVAL --------------------
    lines.append("2. RETRIEVAL EVALUATION")
    lines.append("-" * 78)

    if retrieval_metrics["available"]:
        lines.append("Test cases: " + str(retrieval_metrics["test_cases"]))
        lines.append("")
        lines.append("Selected Retrieval Configuration: " + fmt(retrieval_quality.get("selected_config")))
        lines.append("")

        config_order = ["semantic", "bm25", "hybrid", "hybrid_rerank", "default"]
        seen = set()
        ordered_names = [n for n in config_order if n in retrieval_metrics["configs"]]
        ordered_names += [n for n in retrieval_metrics["configs"] if n not in ordered_names]

        for name in ordered_names:
            vals = retrieval_metrics["configs"][name]
            is_selected = (name == retrieval_quality.get("selected_config"))
            marker = " <== SELECTED / BEST" if is_selected else ""
            lines.append(f"  [{name}]{marker}")
            lines.append("    Precision@5: " + fmt(vals.get("precision")) + "%")
            lines.append("    Recall@5: " + fmt(vals.get("recall")) + "%")
            lines.append("    Hit Rate@5: " + fmt(vals.get("hit_rate")) + "%")
            lines.append("    MRR@5: " + fmt(vals.get("mrr")) + "%")
            lines.append("    NDCG@5: " + fmt(vals.get("ndcg")) + ("%" if vals.get("ndcg") is not None else ""))
            lines.append("    MAP@5: " + fmt(vals.get("map")) + ("%" if vals.get("map") is not None else ""))
            lines.append("    Config Score: " + fmt(vals.get("score")) + "%")
            lines.append("")

        lines.append("Final Retrieval Quality Score: " + fmt(retrieval_quality.get("score")) + "%")
    else:
        lines.append("Retrieval evaluation not available.")

    lines.append("")
    lines.append("Hybrid 70/30 — cross-validation (hybrid_retrieval_evaluation_results.json)")

    if hybrid_metrics["available"]:
        lines.append("  Test cases: " + str(hybrid_metrics["test_cases"]))
        lines.append("  Reranker used: " + str(hybrid_metrics.get("reranker_used")))
        for key, value in hybrid_metrics["metrics"].items():
            lines.append(f"  {key}: {value:.2f}%")
        lines.append("  Cross-check score: " + fmt(hybrid_metrics.get("score")) + "%")
        lines.append("  (Not included in the main retrieval score to avoid double-counting)")
    else:
        lines.append("  Not available.")

    lines.append("")

    # -------------------- GROUNDING --------------------
    lines.append("3. GROUNDING & CITATION")
    lines.append("-" * 78)
    lines.append("Claim Verification Accuracy: " + fmt(claim_metrics.get("test_accuracy")) + "%")
    lines.append("Claim Support Rate: " + fmt(claim_metrics.get("claim_support_rate")) + "%")
    lines.append("Claim Coverage Rate: " + fmt(claim_metrics.get("claim_coverage_rate")) + "%")
    lines.append("Citation Accuracy: " + fmt(citation_metrics.get("citation_accuracy")) + "%")
    lines.append("Strict Faithfulness: " + fmt(faithfulness_metrics.get("strict_rate")) + "%")
    lines.append("Broad Faithfulness: " + fmt(faithfulness_metrics.get("broad_rate")) + "%")
    lines.append("Confidence Gate Accuracy: " + fmt(confidence_metrics.get("accuracy")) + "%")
    lines.append("")
    lines.append("Overall Grounding Score: " + fmt(grounding_score.get("percentage")) + "%")
    lines.append("")

    # -------------------- CLAIM STATISTICS --------------------
    lines.append("4. CLAIM STATISTICS")
    lines.append("-" * 78)
    lines.append("Total tests: " + str(claim_metrics.get("total_tests", 0)))
    lines.append("Passed tests: " + str(claim_metrics.get("passed_tests", 0)))
    lines.append("Total claims: " + str(claim_metrics.get("total_claims", 0)))
    lines.append("Supported claims: " + str(claim_metrics.get("supported_claims", 0)))
    lines.append("Partially supported: " + str(claim_metrics.get("partially_supported_claims", 0)))
    lines.append("Unsupported: " + str(claim_metrics.get("unsupported_claims", 0)))
    lines.append("Claims extracted: " + str(claim_extraction_metrics.get("claims_extracted", 0)))
    lines.append("")

    # -------------------- SAFETY & REFUSAL --------------------
    lines.append("5. SAFETY & REFUSAL")
    lines.append("-" * 78)
    lines.append("Safety Accuracy: " + fmt(safety_metrics.get("accuracy")) + "%")
    lines.append("Safety Test Cases: " + str(safety_metrics.get("test_cases", 0)))
    lines.append("")
    lines.append("Refusal Accuracy: " + fmt(refusal_metrics.get("accuracy")) + "%")
    lines.append("Refusal Test Cases: " + str(refusal_metrics.get("test_cases", 0)))
    lines.append("Selected Refusal Threshold: " + fmt(refusal_metrics.get("selected_threshold")))
    if refusal_metrics.get("available"):
        lines.append("  Precision: " + fmt(refusal_metrics.get("precision")) + "%")
        lines.append("  Recall: " + fmt(refusal_metrics.get("recall")) + "%")
        lines.append("  Correct accepts: " + str(refusal_metrics.get("correct_accept")))
        lines.append("  Correct refusals: " + str(refusal_metrics.get("correct_refuse")))
        lines.append("  False accepts: " + str(refusal_metrics.get("false_accept")))
        lines.append("  False rejects: " + str(refusal_metrics.get("false_reject")))
    else:
        lines.append("  (Ground truth set only — no actual system decisions to score.)")
    lines.append("")
    lines.append("Confidence Gate Accuracy: " + fmt(confidence_metrics.get("accuracy")) + "%")
    lines.append("")

    lines.append("Threshold Experiment (confidence-gate acceptance, supplementary):")
    if threshold_experiment_metrics.get("available"):
        lines.append("  Test cases: " + str(threshold_experiment_metrics.get("test_cases")))
        lines.append("  Selected threshold: " + fmt(threshold_experiment_metrics.get("selected_threshold")))
        lines.append("  Accept precision: " + fmt(threshold_experiment_metrics.get("accept_precision")) + "%")
        lines.append("  Relevant recall: " + fmt(threshold_experiment_metrics.get("relevant_recall")) + "%")
        lines.append("  Acceptance rate: " + fmt(threshold_experiment_metrics.get("acceptance_rate")) + "%")
        if threshold_experiment_metrics.get("all_thresholds_identical"):
            lines.append("  Note: all tested thresholds produced identical accept/reject outcomes.")
    else:
        lines.append("  Not available.")
    lines.append("")

    lines.append("Day 3 Comprehensive Test Suite (supplementary, not double-counted):")
    if day3_test_metrics.get("available"):
        lines.append("  Total tests: " + str(day3_test_metrics.get("total_tests")))
        lines.append("  Passed: " + str(day3_test_metrics.get("passed")))
        lines.append("  Failed: " + str(day3_test_metrics.get("failed")))
        lines.append("  Pass rate: " + fmt(day3_test_metrics.get("pass_rate")) + "%")
    else:
        lines.append("  Not available.")
    lines.append("")

    # -------------------- LATENCY --------------------
    lines.append("6. PERFORMANCE / LATENCY")
    lines.append("-" * 78)
    lines.append("Test cases: " + str(latency_metrics.get("test_cases", 0)))
    lines.append("Average: " + fmt(latency_metrics.get("average_ms")) + " ms")
    lines.append("Median: " + fmt(latency_metrics.get("median_ms")) + " ms")
    lines.append("P95: " + fmt(latency_metrics.get("p95_ms")) + " ms")
    lines.append("Min: " + fmt(latency_metrics.get("min_ms")) + " ms")
    lines.append("Max: " + fmt(latency_metrics.get("max_ms")) + " ms")
    lines.append("")

    # -------------------- EVALUATION DEPTH --------------------
    lines.append("7. EVALUATION DEPTH")
    lines.append("-" * 78)
    lines.append("Evaluation Cases: " + str(dataset_depth.get("test_cases", 0)))
    lines.append("Recommended Minimum: " + str(dataset_depth.get("recommended_minimum", 20)))
    lines.append("Completion: " + fmt(dataset_depth.get("completion_percentage")) + "%")
    lines.append(
        "Target: " + ("MET" if dataset_depth.get("meets_20_question_target") else "NOT MET")
    )
    lines.append("")

    # -------------------- FAILURE ANALYSIS --------------------
    lines.append("8. FAILURE ANALYSIS")
    lines.append("-" * 78)
    lines.append("Claim-related issues: " + str(len(claim_failures)))
    lines.append("Faithfulness issues: " + str(len(faithfulness_failures)))
    lines.append("Retrieval issues: " + str(len(retrieval_failures)))
    lines.append("Citation issues: " + str(len(citation_failures)))
    lines.append("Safety issues: " + str(len(safety_failures)))
    lines.append("")

    if claim_failures:
        lines.append("Claim failures:")
        for failure in claim_failures:
            lines.append(f'  - {failure["id"]}: {failure["status"]} — {failure["question"]}')

    if faithfulness_failures:
        lines.append("")
        lines.append("Faithfulness failures:")
        for failure in faithfulness_failures:
            lines.append(
                f'  - {failure["id"]}: {failure["status"]} — '
                f'{failure["question"]} — Score={failure["score"]}'
            )

    if retrieval_failures:
        lines.append("")
        lines.append("Retrieval failures:")
        for failure in retrieval_failures:
            lines.append(f'  - {failure["id"]}: {failure["status"]} — {failure["question"]}')

    if citation_failures:
        lines.append("")
        lines.append("Citation failures:")
        for failure in citation_failures:
            lines.append(f'  - {failure["id"]}: {failure["status"]} — {failure["question"]}')

    if safety_failures:
        lines.append("")
        lines.append("Safety failures:")
        for failure in safety_failures:
            lines.append(f'  - {failure["id"]}: {failure["status"]} — {failure["question"]}')

    lines.append("")

    # -------------------- RUBRIC --------------------
    lines.append("9. JUDGE-ORIENTED RUBRIC EVIDENCE")
    lines.append("-" * 78)

    for key, label in [
        ("retrieval_quality_30_percent", "Retrieval Quality — 30%"),
        ("grounding_citation_25_percent", "Grounding & Citation — 25%"),
        ("system_architecture_15_percent", "System Architecture — 15%"),
        ("evaluation_depth_15_percent", "Evaluation Depth — 15%"),
        ("safety_ux_15_percent", "Safety & UX — 15%"),
    ]:
        section = rubric_summary[key]
        lines.append(label)
        score = section.get("measured_score")
        lines.append("  Score: " + fmt(score) + ("%" if score is not None else ""))
        lines.append("  Status: " + str(section.get("status", "")))

    lines.append("")

    # -------------------- MEASURABLE INDICATOR --------------------
    lines.append("10. MEASURABLE PROJECT INDICATOR")
    lines.append("-" * 78)
    lines.append("Score: " + fmt(measurable_score.get("score")) + "%")
    lines.append(
        "Available Weight: " + str(measurable_score.get("max_available_weight", "N/A"))
    )
    lines.append(
        "Excluded Weight (Architecture, judge-reviewed): "
        + str(measurable_score.get("official_architecture_weight", 15))
    )
    lines.append("Note: " + str(measurable_score.get("note", "")))
    lines.append("")

    # -------------------- ARTIFACTS --------------------
    lines.append("11. EVALUATION ARTIFACT AVAILABILITY")
    lines.append("-" * 78)
    for name, path in FILES.items():
        status = "AVAILABLE" if loaded.get(name) is not None else "MISSING"
        lines.append(f"{name}: {status}")
    lines.append("")

    # -------------------- JUDGE NOTE --------------------
    lines.append("12. IMPORTANT JUDGING NOTE")
    lines.append("-" * 78)
    lines.append("This report provides quantitative evidence for the judging rubric.")
    lines.append("It is NOT the official hackathon score.")
    lines.append("System Architecture (15%) requires judge review.")
    lines.append(
        "Live UX, clinical disclaimer, citation display, pipeline architecture, "
        "modularity, and demo quality must be demonstrated separately."
    )
    lines.append("")
    lines.append("=" * 78)
    lines.append("END OF COMPREHENSIVE FINAL EVALUATION")
    lines.append("=" * 78)

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("PULMO GUIDE")
    print("DAY 4 — COMPREHENSIVE FINAL EVALUATION")
    print("=" * 78)
    print()
    print("Deterministic evaluation.")
    print("No LLM required.")
    print()

    loaded = load_all_files()

    # -------------------- INDIVIDUAL METRICS --------------------
    claim_metrics = evaluate_claim_verification(loaded["claim_verification"])
    citation_metrics = evaluate_citations(loaded["citation_check"])
    faithfulness_metrics = evaluate_faithfulness(loaded["faithfulness"])
    confidence_metrics = evaluate_confidence_gate(loaded["confidence_gate"])

    retrieval_metrics = evaluate_retrieval(loaded["retrieval"])
    hybrid_metrics = evaluate_hybrid_retrieval(loaded["hybrid_retrieval"])

    safety_metrics = evaluate_safety(loaded["day3_safety"])
    day3_test_metrics = evaluate_day3_test_suite(loaded["day3_test"])
    refusal_metrics = choose_refusal_metrics(loaded)
    threshold_experiment_metrics = evaluate_threshold_experiment(loaded["threshold_experiment"])

    latency_metrics = evaluate_latency(loaded["latency"])
    claim_extraction_metrics = evaluate_claim_extraction(loaded["claim_extraction"])
    dataset_depth = evaluate_dataset_depth(loaded["evaluation_set"])

    # -------------------- AGGREGATED SCORES --------------------
    grounding_score = calculate_grounding_score(
        claim_metrics, citation_metrics, faithfulness_metrics, confidence_metrics,
    )
    retrieval_quality = calculate_retrieval_quality(retrieval_metrics)
    safety_score = calculate_safety_score(safety_metrics, refusal_metrics)
    measurable_score = calculate_measurable_score(
        retrieval_quality, grounding_score, safety_score, dataset_depth,
    )
    rubric_summary = build_rubric_summary(
        retrieval_quality, grounding_score, safety_score, dataset_depth,
    )

    # -------------------- FAILURE ANALYSIS --------------------
    claim_failures = analyze_claim_failures(loaded["claim_verification"])
    faithfulness_failures = analyze_faithfulness_failures(loaded["faithfulness"])
    retrieval_failures = analyze_retrieval_failures(loaded["retrieval"])
    citation_failures = analyze_generic_failures(
        loaded["citation_check"],
        {"FAIL", "FAILED", "INVALID", "UNSUPPORTED", "INCORRECT"},
        "citation",
    )
    safety_failures = analyze_generic_failures(
        loaded["day3_safety"],
        {"FAIL", "FAILED", "UNSAFE", "INCORRECT"},
        "safety",
    )

    # -------------------- CONSOLE OUTPUT --------------------
    print()
    print("COMPREHENSIVE RESULTS")
    print("-" * 78)
    print("Claim Verification Accuracy: " + fmt(claim_metrics.get("test_accuracy")) + "%")
    print("Claim Support Rate: " + fmt(claim_metrics.get("claim_support_rate")) + "%")
    print("Citation Accuracy: " + fmt(citation_metrics.get("citation_accuracy")) + "%")
    print("Strict Faithfulness Rate: " + fmt(faithfulness_metrics.get("strict_rate")) + "%")
    print("Broad Faithfulness Rate: " + fmt(faithfulness_metrics.get("broad_rate")) + "%")
    print("Confidence Gate Accuracy: " + fmt(confidence_metrics.get("accuracy")) + "%")
    print("Retrieval Quality Score: " + fmt(retrieval_quality.get("score")) + "%")
    print("Safety Accuracy: " + fmt(safety_metrics.get("accuracy")) + "%")
    print("Refusal Accuracy: " + fmt(refusal_metrics.get("accuracy")) + "%")
    print("Safety / UX Evidence Score: " + fmt(safety_score.get("score")) + "%")
    print()
    print("OVERALL GROUNDING SCORE: " + fmt(grounding_score.get("percentage")) + "%")
    print("MEASURABLE PROJECT INDICATOR: " + fmt(measurable_score.get("score")) + "%")
    print()

    print("EVALUATION DEPTH")
    print("-" * 78)
    print("Current test cases: " + str(dataset_depth.get("test_cases", 0)))
    if dataset_depth.get("meets_20_question_target"):
        print("OK: 20+ question target MET.")
    else:
        print("WARNING: Less than 20 evaluation questions.")
    print()

    print("FAILURE ANALYSIS")
    print("-" * 78)
    print("Claim issues: " + str(len(claim_failures)))
    print("Faithfulness issues: " + str(len(faithfulness_failures)))
    print("Retrieval issues: " + str(len(retrieval_failures)))
    print("Citation issues: " + str(len(citation_failures)))
    print("Safety issues: " + str(len(safety_failures)))
    print()

    # -------------------- BUILD REPORT --------------------
    report = build_report(
        loaded=loaded,
        claim_metrics=claim_metrics,
        citation_metrics=citation_metrics,
        faithfulness_metrics=faithfulness_metrics,
        confidence_metrics=confidence_metrics,
        retrieval_metrics=retrieval_metrics,
        hybrid_metrics=hybrid_metrics,
        safety_metrics=safety_metrics,
        refusal_metrics=refusal_metrics,
        threshold_experiment_metrics=threshold_experiment_metrics,
        day3_test_metrics=day3_test_metrics,
        latency_metrics=latency_metrics,
        claim_extraction_metrics=claim_extraction_metrics,
        grounding_score=grounding_score,
        retrieval_quality=retrieval_quality,
        safety_score=safety_score,
        measurable_score=measurable_score,
        dataset_depth=dataset_depth,
        rubric_summary=rubric_summary,
        claim_failures=claim_failures,
        faithfulness_failures=faithfulness_failures,
        retrieval_failures=retrieval_failures,
        citation_failures=citation_failures,
        safety_failures=safety_failures,
    )

    # -------------------- FINAL JSON --------------------
    output = {
        "project": "Pulmo Guide",
        "evaluation": "comprehensive_final_evaluation",
        "day": 4,
        "method": "deterministic",
        "langchain_used": False,
        "llm_required": False,

        "metrics": {
            "claim_verification": claim_metrics,
            "claim_extraction": claim_extraction_metrics,
            "citation_check": citation_metrics,
            "faithfulness": faithfulness_metrics,
            "confidence_gate": confidence_metrics,
            "retrieval": retrieval_metrics,
            "hybrid_retrieval": hybrid_metrics,
            "safety": safety_metrics,
            "refusal": refusal_metrics,
            "threshold_experiment": threshold_experiment_metrics,
            "day3_test_suite": day3_test_metrics,
            "latency": latency_metrics,
        },

        "aggregated_scores": {
            "grounding": grounding_score,
            "retrieval_quality": retrieval_quality,
            "safety_ux": safety_score,
            "measurable_project_indicator": measurable_score,
        },

        "judge_rubric": rubric_summary,

        "failure_analysis": {
            "claim_failures": claim_failures,
            "faithfulness_failures": faithfulness_failures,
            "retrieval_failures": retrieval_failures,
            "citation_failures": citation_failures,
            "safety_failures": safety_failures,
        },

        "evaluation_depth": dataset_depth,

        "available_evaluation_files": {
            name: loaded[name] is not None for name in FILES
        },

        "architecture_evaluation": {
            "automated_score": None,
            "status": "JUDGE_REVIEW_REQUIRED",
            "reason": (
                "System architecture quality requires human inspection of "
                "modularity, pipeline separation, architecture diagram, code "
                "organization, and implementation structure."
            ),
        },

        "note": (
            "This report aggregates quantitative evaluation evidence. "
            "It does not represent the official hackathon judge score."
        ),
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    with open(FINAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    FINAL_REPORT_FILE.write_text(report, encoding="utf-8")

    # -------------------- FINAL OUTPUT --------------------
    print("=" * 78)
    print("FINAL RESULT")
    print("=" * 78)
    print("Overall Grounding Score: " + fmt(grounding_score.get("percentage")) + "%")
    print("Retrieval Quality Score: " + fmt(retrieval_quality.get("score")) + "%")
    print("Safety / UX Evidence Score: " + fmt(safety_score.get("score")) + "%")
    print("Measurable Project Indicator: " + fmt(measurable_score.get("score")) + "%")
    print("Evaluation Cases: " + str(dataset_depth.get("test_cases", 0)))
    print()
    print("JSON saved to:")
    print(FINAL_RESULTS_FILE)
    print()
    print("Report saved to:")
    print(FINAL_REPORT_FILE)
    print()
    print("Comprehensive final evaluation completed successfully.")


if __name__ == "__main__":
    main()
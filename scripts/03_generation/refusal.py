"""
============================================================
PULMO GUIDE
DAY 3 - REFUSAL / CONFIDENCE GATE
============================================================

Pipeline:

    User Query
        |
        v
    Hybrid Retrieval (70/30)       [LOCKED]
        |
        v
    Refusal / Confidence Gate      [THIS MODULE]
        |
        +----> REFUSE
        |
        v
    Grounded Prompt                [NEXT]
        |
        v
    LLM                            [NEXT]
        |
        v
    Answer + Citation              [NEXT]

IMPORTANT:
- Retrieval is LOCKED.
- This module does NOT modify retrieval.py.
- This module does NOT modify retrieval_config.py.
- This module does NOT modify ChromaDB.
- This module does NOT call an LLM.
- This module only decides whether generation is allowed.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class RefusalConfig:
    """
    Configuration for the refusal gate.

    NOTE:
    hybrid_score is NOT a calibrated probability.

    The current threshold (0.80) is only a conservative
    starting heuristic based on the Day 3 experiment.
    """

    # Current experimental starting point.
    hybrid_score_threshold: float = 0.80

    # Minimum number of retrieved chunks required.
    min_retrieved_chunks: int = 1

    # Reject empty queries.
    require_non_empty_query: bool = True


DEFAULT_CONFIG = RefusalConfig()


# ============================================================
# RESULT BUILDER
# ============================================================

def _build_result(
    decision: str,
    reason: str,
    confidence: Optional[float],
    threshold: float,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a standardized refusal-gate result.
    """

    return {
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "threshold": threshold,

        # Important:
        # hybrid_score is a retrieval score, not a probability.
        "confidence_is_calibrated": False,

        # True means the request may continue to generation.
        "generation_allowed": decision == "accept",

        # Retrieved evidence is passed unchanged to later stages.
        "retrieved_chunks": retrieved_chunks,
    }


# ============================================================
# REFUSAL / CONFIDENCE GATE
# ============================================================

def check_refusal(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    config: Optional[RefusalConfig] = None,
) -> Dict[str, Any]:
    """
    Decide whether the system should:

        ACCEPT -> continue to grounded generation

    or

        REFUSE -> do not call the LLM

    Parameters
    ----------
    query:
        User's original question.

    retrieved_results:
        Output from the locked Hybrid Retrieval.

        Expected structure:

        [
            {
                "hybrid_rank": 1,
                "chunk_id": "...",
                "text": "...",
                "metadata": {...},
                "semantic_score": ...,
                "semantic_normalized": ...,
                "bm25_score": ...,
                "bm25_normalized": ...,
                "hybrid_score": ...
            },
            ...
        ]

    config:
        Optional RefusalConfig.

    Returns
    -------
    dict
        Standardized refusal decision.
    """

    cfg = config or DEFAULT_CONFIG

    # --------------------------------------------------------
    # 1. Empty query guard
    # --------------------------------------------------------

    if cfg.require_non_empty_query:

        if not query or not query.strip():

            return _build_result(
                decision="refuse",
                reason="Empty or missing query.",
                confidence=None,
                threshold=cfg.hybrid_score_threshold,
                retrieved_chunks=retrieved_results or [],
            )

    # --------------------------------------------------------
    # 2. Retrieval failure / no evidence
    # --------------------------------------------------------

    if not retrieved_results:

        return _build_result(
            decision="refuse",
            reason=(
                "No evidence was retrieved from the NICE NG122 "
                "knowledge base."
            ),
            confidence=None,
            threshold=cfg.hybrid_score_threshold,
            retrieved_chunks=[],
        )

    # --------------------------------------------------------
    # 3. Minimum retrieval check
    # --------------------------------------------------------

    if len(retrieved_results) < cfg.min_retrieved_chunks:

        return _build_result(
            decision="refuse",
            reason=(
                "Insufficient retrieved evidence to answer safely."
            ),
            confidence=None,
            threshold=cfg.hybrid_score_threshold,
            retrieved_chunks=retrieved_results,
        )

    # --------------------------------------------------------
    # 4. Get top-ranked result
    # --------------------------------------------------------

    # retrieval.py already sorts by hybrid_score descending.
    top_chunk = retrieved_results[0]

    confidence = top_chunk.get("hybrid_score")

    # --------------------------------------------------------
    # 5. Missing score guard
    # --------------------------------------------------------

    if confidence is None:

        return _build_result(
            decision="refuse",
            reason=(
                "The retrieved result does not contain a valid "
                "hybrid_score."
            ),
            confidence=None,
            threshold=cfg.hybrid_score_threshold,
            retrieved_chunks=retrieved_results,
        )

    # --------------------------------------------------------
    # 6. Validate score
    # --------------------------------------------------------

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):

        return _build_result(
            decision="refuse",
            reason="Invalid hybrid_score returned by retrieval.",
            confidence=None,
            threshold=cfg.hybrid_score_threshold,
            retrieved_chunks=retrieved_results,
        )

    # --------------------------------------------------------
    # 7. Threshold decision
    # --------------------------------------------------------

    if confidence >= cfg.hybrid_score_threshold:

        return _build_result(
            decision="accept",
            reason=(
                f"Retrieved evidence passed the current refusal "
                f"threshold ({confidence:.4f} >= "
                f"{cfg.hybrid_score_threshold:.2f})."
            ),
            confidence=confidence,
            threshold=cfg.hybrid_score_threshold,
            retrieved_chunks=retrieved_results,
        )

    # --------------------------------------------------------
    # 8. Refuse
    # --------------------------------------------------------

    return _build_result(
        decision="refuse",
        reason=(
            f"Retrieved evidence did not pass the current refusal "
            f"threshold ({confidence:.4f} < "
            f"{cfg.hybrid_score_threshold:.2f}). "
            "The information is not sufficiently supported by "
            "the NICE NG122 knowledge base."
        ),
        confidence=confidence,
        threshold=cfg.hybrid_score_threshold,
        retrieved_chunks=retrieved_results,
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def should_generate(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    config: Optional[RefusalConfig] = None,
) -> bool:
    """
    Simple helper used by the future generation pipeline.

    Returns:
        True  -> generation is allowed
        False -> system should refuse
    """

    result = check_refusal(
        query=query,
        retrieved_results=retrieved_results,
        config=config,
    )

    return result["generation_allowed"]


def get_refusal_message() -> str:
    """
    Standard safe refusal message.

    This is intentionally short and does not invent medical content.
    """

    return (
        "I’m sorry, but I couldn’t find sufficient information "
        "in the available NICE lung cancer guideline to answer "
        "this question safely."
    )


# ============================================================
# MANUAL SMOKE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PULMO GUIDE - REFUSAL GATE SMOKE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Case 1: Strong retrieved evidence
    # --------------------------------------------------------

    fake_in_kb_result = [
        {
            "hybrid_rank": 1,
            "chunk_id": "chunk_001",
            "text": (
                "Offer contrast-enhanced brain MRI for people "
                "with stage 3 NSCLC."
            ),
            "metadata": {
                "section": "Further staging",
                "page_start": 14,
                "page_end": 14,
            },
            "semantic_score": 0.90,
            "semantic_normalized": 0.95,
            "bm25_score": 8.2,
            "bm25_normalized": 0.88,
            "hybrid_score": 0.91,
        }
    ]

    result = check_refusal(
        "What imaging is offered in stage 3 NSCLC?",
        fake_in_kb_result,
    )

    print("\nCASE 1 - IN-KB")
    print(result)

    # --------------------------------------------------------
    # Case 2: Weak retrieved evidence
    # --------------------------------------------------------

    fake_weak_result = [
        {
            "hybrid_rank": 1,
            "chunk_id": "chunk_002",
            "text": "Unrelated lung cancer information.",
            "metadata": {
                "section": "General",
                "page_start": 1,
                "page_end": 1,
            },
            "hybrid_score": 0.42,
        }
    ]

    result = check_refusal(
        "What is the capital of France?",
        fake_weak_result,
    )

    print("\nCASE 2 - WEAK / OUT-OF-KB")
    print(result)

    # --------------------------------------------------------
    # Case 3: Empty query
    # --------------------------------------------------------

    result = check_refusal(
        "",
        fake_in_kb_result,
    )

    print("\nCASE 3 - EMPTY QUERY")
    print(result)

    # --------------------------------------------------------
    # Case 4: No retrieval results
    # --------------------------------------------------------

    result = check_refusal(
        "Some question",
        [],
    )

    print("\nCASE 4 - NO RETRIEVAL RESULTS")
    print(result)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETED")
    print("=" * 60)
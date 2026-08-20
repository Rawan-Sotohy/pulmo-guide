"""
============================================================
PULMO GUIDE
DAY 3 - EVIDENCE DECISION / REFUSAL GATE
============================================================

FINAL RETRIEVAL CONFIGURATION:

    Semantic Retrieval = 70%
    BM25 Retrieval     = 30%
    Final Top K        = 5
    Reranker           = OFF

Pipeline:

    User Query
        |
        v
    Safety Decision
        |
        +----> REFUSE
        |
        v
    Hybrid Retrieval
    70% Semantic + 30% BM25
        |
        v
    Final Top 5
        |
        v
    THIS MODULE
        |
        +----> INSUFFICIENT
        |
        +----> WEAK
        |
        +----> PARTIAL
        |
        +----> STRONG
        |
        v
    Citation Builder
        |
        v
    Grounded Prompt
        |
        v
    LLM

IMPORTANT:

- Retrieval is LOCKED.
- No reranker is used.
- This module does NOT modify retrieval.py.
- This module does NOT modify ChromaDB.
- This module does NOT call an LLM.
- This module does NOT create citations.
- This module only evaluates retrieved evidence.
============================================================
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


# ============================================================
# FINAL RETRIEVAL CONFIGURATION
# ============================================================

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30
FINAL_TOP_K = 5
USE_RERANKER = False


# ============================================================
# EVIDENCE LEVELS
# ============================================================

class EvidenceLevel:

    INSUFFICIENT = "insufficient"
    WEAK = "weak"
    PARTIAL = "partial"
    STRONG = "strong"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class RefusalConfig:
    """
    Evidence decision thresholds.

    IMPORTANT:

    hybrid_score is a retrieval score.
    It is NOT a calibrated probability.

    Threshold policy:

        < 0.65
            INSUFFICIENT

        0.65 <= score < 0.75
            WEAK

        0.75 <= score < 0.85
            PARTIAL

        score >= 0.85
            STRONG
    """

    insufficient_threshold: float = 0.65

    weak_threshold: float = 0.75

    strong_threshold: float = 0.85

    min_retrieved_chunks: int = 1

    require_non_empty_query: bool = True


DEFAULT_CONFIG = RefusalConfig()


# ============================================================
# RESULT BUILDER
# ============================================================

def _build_result(
    decision: str,
    reason: str,
    confidence: Optional[float],
    config: RefusalConfig,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Only keep final Top 5.
    # --------------------------------------------------------

    retrieved_chunks = (
        retrieved_chunks[:FINAL_TOP_K]
        if retrieved_chunks
        else []
    )

    return {

        "decision": decision,

        "evidence_level": decision,

        "reason": reason,

        # Retrieval score only.
        "confidence": confidence,

        # Explicitly NOT a probability.
        "confidence_is_calibrated": False,

        # Final retrieval configuration.
        "retrieval_configuration": {

            "semantic_weight":
                SEMANTIC_WEIGHT,

            "bm25_weight":
                BM25_WEIGHT,

            "final_top_k":
                FINAL_TOP_K,

            "reranker":
                USE_RERANKER,
        },

        "thresholds": {

            "insufficient":
                config.insufficient_threshold,

            "weak":
                config.weak_threshold,

            "strong":
                config.strong_threshold,
        },

        # ----------------------------------------------------
        # Generation policy
        # ----------------------------------------------------

        "generation_allowed": decision in {

            EvidenceLevel.WEAK,

            EvidenceLevel.PARTIAL,

            EvidenceLevel.STRONG,
        },

        "retrieved_chunks":
            retrieved_chunks,
    }


# ============================================================
# EVIDENCE DECISION
# ============================================================

def check_refusal(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    config: Optional[RefusalConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate retrieved evidence.

    Returns one of:

        insufficient
        weak
        partial
        strong
    """

    cfg = config or DEFAULT_CONFIG


    # ========================================================
    # 1. EMPTY QUERY
    # ========================================================

    if cfg.require_non_empty_query:

        if not query or not query.strip():

            return _build_result(

                decision=EvidenceLevel.INSUFFICIENT,

                reason="Empty or missing query.",

                confidence=None,

                config=cfg,

                retrieved_chunks=
                    retrieved_results or [],
            )


    # ========================================================
    # 2. NO EVIDENCE
    # ========================================================

    if not retrieved_results:

        return _build_result(

            decision=EvidenceLevel.INSUFFICIENT,

            reason=(
                "No relevant evidence was retrieved "
                "from the indexed clinical knowledge base."
            ),

            confidence=None,

            config=cfg,

            retrieved_chunks=[],
        )


    # ========================================================
    # 3. KEEP FINAL TOP 5
    # ========================================================

    retrieved_results = retrieved_results[
        :FINAL_TOP_K
    ]


    # ========================================================
    # 4. MINIMUM RETRIEVAL CHECK
    # ========================================================

    if len(retrieved_results) < cfg.min_retrieved_chunks:

        return _build_result(

            decision=EvidenceLevel.INSUFFICIENT,

            reason=(
                "Insufficient retrieved evidence "
                "to answer safely."
            ),

            confidence=None,

            config=cfg,

            retrieved_chunks=retrieved_results,
        )


    # ========================================================
    # 5. TOP RESULT
    # ========================================================

    top_chunk = retrieved_results[0]

    score = top_chunk.get(
        "hybrid_score"
    )


    # ========================================================
    # 6. MISSING SCORE
    # ========================================================

    if score is None:

        return _build_result(

            decision=EvidenceLevel.INSUFFICIENT,

            reason=(
                "The retrieved result does not contain "
                "a valid hybrid_score."
            ),

            confidence=None,

            config=cfg,

            retrieved_chunks=retrieved_results,
        )


    # ========================================================
    # 7. VALIDATE SCORE
    # ========================================================

    try:

        score = float(score)

    except (TypeError, ValueError):

        return _build_result(

            decision=EvidenceLevel.INSUFFICIENT,

            reason=(
                "Invalid hybrid_score returned "
                "by retrieval."
            ),

            confidence=None,

            config=cfg,

            retrieved_chunks=retrieved_results,
        )


    # ========================================================
    # 8. INVALID NUMERIC RANGE
    # ========================================================

    if not 0.0 <= score <= 1.0:

        return _build_result(

            decision=EvidenceLevel.INSUFFICIENT,

            reason=(
                f"Invalid hybrid_score range: {score:.4f}."
            ),

            confidence=None,

            config=cfg,

            retrieved_chunks=retrieved_results,
        )


    # ========================================================
    # 9. INSUFFICIENT
    # ========================================================

    if score < cfg.insufficient_threshold:

        return _build_result(

            decision=EvidenceLevel.INSUFFICIENT,

            reason=(
                f"Retrieved evidence score ({score:.4f}) "
                f"is below the minimum evidence threshold "
                f"({cfg.insufficient_threshold:.2f})."
            ),

            confidence=score,

            config=cfg,

            retrieved_chunks=retrieved_results,
        )


    # ========================================================
    # 10. WEAK
    # ========================================================

    if score < cfg.weak_threshold:

        return _build_result(

            decision=EvidenceLevel.WEAK,

            reason=(
                f"Evidence score ({score:.4f}) indicates "
                f"weak retrieval evidence. "
                f"Only explicitly supported information "
                f"may be used."
            ),

            confidence=score,

            config=cfg,

            retrieved_chunks=retrieved_results,
        )


    # ========================================================
    # 11. PARTIAL
    # ========================================================

    if score < cfg.strong_threshold:

        return _build_result(

            decision=EvidenceLevel.PARTIAL,

            reason=(
                f"Evidence score ({score:.4f}) indicates "
                f"partial evidence. "
                f"Only the supported part of the question "
                f"may be answered."
            ),

            confidence=score,

            config=cfg,

            retrieved_chunks=retrieved_results,
        )


    # ========================================================
    # 12. STRONG
    # ========================================================

    return _build_result(

        decision=EvidenceLevel.STRONG,

        reason=(
            f"Evidence score ({score:.4f}) meets "
            f"the strong evidence threshold "
            f"({cfg.strong_threshold:.2f})."
        ),

        confidence=score,

        config=cfg,

        retrieved_chunks=retrieved_results,
    )


# ============================================================
# HELPER
# ============================================================

def should_generate(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    config: Optional[RefusalConfig] = None,
) -> bool:

    result = check_refusal(

        query=query,

        retrieved_results=retrieved_results,

        config=config,
    )

    return result["generation_allowed"]


# ============================================================
# REFUSAL MESSAGE
# ============================================================

def get_refusal_message() -> str:

    return (
        "I couldn't find enough relevant evidence in the "
        "indexed guidelines to answer this question confidently."
    )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 75)

    print(
        "PULMO GUIDE - DAY 3 "
        "EVIDENCE DECISION TEST"
    )

    print("=" * 75)

    print("\nFinal Retrieval Configuration:")

    print(
        f"Semantic Weight : {SEMANTIC_WEIGHT}"
    )

    print(
        f"BM25 Weight     : {BM25_WEIGHT}"
    )

    print(
        f"Final Top K     : {FINAL_TOP_K}"
    )

    print(
        f"Reranker        : {USE_RERANKER}"
    )


    test_scores = [

        0.40,

        0.68,

        0.78,

        0.91,
    ]


    for score in test_scores:

        fake_result = [

            {

                "hybrid_rank": 1,

                "chunk_id": "test_chunk",

                "text":
                    "Test retrieved clinical evidence.",

                "metadata": {

                    "document_name":
                        "NICE NG122",

                    "section":
                        "Test Section",

                    "page_start":
                        10,

                    "page_end":
                        10,

                    "citation":
                        "[NICE NG122, Test Section, Page 10]",
                },

                "hybrid_score":
                    score,
            }
        ]


        result = check_refusal(

            query=
                "What are the symptoms of lung cancer?",

            retrieved_results=
                fake_result,
        )


        print("\n" + "-" * 75)

        print(
            "SCORE:",
            score
        )

        print(
            "DECISION:",
            result["decision"]
        )

        print(
            "GENERATION ALLOWED:",
            result["generation_allowed"]
        )

        print(
            "REASON:",
            result["reason"]
        )


    # ========================================================
    # NO EVIDENCE
    # ========================================================

    result = check_refusal(

        query=
            "What are the symptoms of lung cancer?",

        retrieved_results=[],
    )


    print("\n" + "-" * 75)

    print("NO EVIDENCE")

    print(
        "DECISION:",
        result["decision"]
    )

    print(
        "GENERATION ALLOWED:",
        result["generation_allowed"]
    )

    print(
        "REASON:",
        result["reason"]
    )


    print("\n" + "=" * 75)

    print(
        "EVIDENCE DECISION TEST COMPLETED"
    )

    print("=" * 75)
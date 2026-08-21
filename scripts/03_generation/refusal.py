"""
============================================================
PULMO GUIDE
DAY 3 - EVIDENCE DECISION / REFUSAL GATE
============================================================

ROLE:
    Evaluate whether retrieved evidence is strong enough
    to allow grounded answer generation.

IMPORTANT:

- Retrieval is LOCKED.
- Semantic Retrieval = 70%
- BM25 Retrieval     = 30%
- Final Top K        = 5
- Reranker           = OFF

This module:

    DOES:
        - Validate retrieved evidence
        - Evaluate evidence quality
        - Decide whether generation is allowed
        - Distinguish Core vs Patient evidence
        - Prevent weak retrieval from being treated as strong

    DOES NOT:
        - Modify retrieval.py
        - Modify ChromaDB
        - Call an LLM
        - Create citations
        - Perform retrieval
        - Diagnose the patient

============================================================
PIPELINE
============================================================

User Query
    |
    v
Safety
    |
    +----> REFUSE
    |
    v
Hybrid Retrieval
    |
    +---- 70% Semantic
    +---- 30% BM25
    |
    v
Top 5
    |
    v
THIS MODULE
    |
    +----> INSUFFICIENT
    +----> WEAK
    +----> PARTIAL
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
    Evidence-quality policy.

    IMPORTANT:

    hybrid_score is NOT used as an absolute confidence
    because retrieval.py applies Min-Max normalization
    across the retrieved candidate set.

    Therefore this module primarily evaluates:

        1. Raw semantic similarity
        2. Relative hybrid retrieval quality
        3. BM25 support
        4. Evidence availability

    Threshold policy:

        Semantic score < 0.50
            -> INSUFFICIENT

        Semantic score >= 0.50
        but < 0.65
            -> WEAK

        Semantic score >= 0.65
        but < 0.75
            -> PARTIAL

        Semantic score >= 0.75
            -> STRONG

    These are evidence heuristics, NOT probabilities.
    """

    # --------------------------------------------------------
    # Raw semantic similarity thresholds
    # --------------------------------------------------------

    semantic_insufficient_threshold: float = 0.50

    semantic_weak_threshold: float = 0.65

    semantic_strong_threshold: float = 0.75

    # --------------------------------------------------------
    # Relative hybrid support
    #
    # hybrid_score comes from Min-Max normalization in
    # retrieval.py, so these thresholds are only used as
    # supporting evidence, NOT as confidence.
    # --------------------------------------------------------

    hybrid_support_threshold: float = 0.30

    strong_hybrid_support_threshold: float = 0.60

    # --------------------------------------------------------
    # BM25 support
    # --------------------------------------------------------

    bm25_support_threshold: float = 0.10

    # --------------------------------------------------------
    # Minimum retrieved chunks
    # --------------------------------------------------------

    min_retrieved_chunks: int = 1

    require_non_empty_query: bool = True


DEFAULT_CONFIG = RefusalConfig()


# ============================================================
# SOURCE DETECTION
# ============================================================

def _detect_source_type(
    retrieved_chunks: List[Dict[str, Any]]
) -> str:

    """
    Detect whether retrieved evidence comes from:

        Core
        Patient
        Mixed
        Unknown
    """

    sources = set()

    for chunk in retrieved_chunks:

        metadata = chunk.get(
            "metadata",
            {}
        )

        source = metadata.get(
            "source_type"
        )

        if source:

            sources.add(
                str(source).lower()
            )

    if not sources:

        return "unknown"

    if sources == {"core"}:

        return "core"

    if sources == {"patient"}:

        return "patient"

    if "core" in sources and "patient" in sources:

        return "mixed"

    return "unknown"


# ============================================================
# RESULT BUILDER
# ============================================================

def _build_result(
    decision: str,
    reason: str,
    confidence: Optional[float],
    config: RefusalConfig,
    retrieved_chunks: List[Dict[str, Any]],
    source_type: str = "unknown",
    evidence_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    retrieved_chunks = (
        retrieved_chunks[:FINAL_TOP_K]
        if retrieved_chunks
        else []
    )

    return {

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        "decision":
            decision,

        "evidence_level":
            decision,

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        "reason":
            reason,

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------
        #
        # This is an evidence-quality score only.
        # It is NOT a probability.
        #

        "confidence":
            confidence,

        "confidence_is_calibrated":
            False,

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        "source_type":
            source_type,

        # ----------------------------------------------------
        # Retrieval configuration
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Evidence thresholds
        # ----------------------------------------------------

        "thresholds": {

            "semantic_insufficient":
                config.semantic_insufficient_threshold,

            "semantic_weak":
                config.semantic_weak_threshold,

            "semantic_strong":
                config.semantic_strong_threshold,

            "hybrid_support":
                config.hybrid_support_threshold,

            "strong_hybrid_support":
                config.strong_hybrid_support_threshold,

            "bm25_support":
                config.bm25_support_threshold,
        },

        # ----------------------------------------------------
        # Evidence metrics
        # ----------------------------------------------------

        "evidence_metrics":
            evidence_metrics or {},

        # ----------------------------------------------------
        # Generation policy
        # ----------------------------------------------------

        "generation_allowed":
            decision in {

                EvidenceLevel.WEAK,

                EvidenceLevel.PARTIAL,

                EvidenceLevel.STRONG,
            },

        # ----------------------------------------------------
        # Retrieved evidence
        # ----------------------------------------------------

        "retrieved_chunks":
            retrieved_chunks,
    }


# ============================================================
# SAFE FLOAT
# ============================================================

def _safe_float(
    value
) -> Optional[float]:

    try:

        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return None


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

    Returns:

        insufficient
        weak
        partial
        strong

    IMPORTANT:

    The decision is NOT based only on hybrid_score.

    Because hybrid_score is Min-Max normalized inside
    retrieval.py, it is relative to the current candidate set.

    The main evidence signal is the raw semantic similarity
    returned by retrieval.py as:

        semantic_score

    BM25 and hybrid scores are supporting signals.
    """

    cfg = config or DEFAULT_CONFIG


    # ========================================================
    # 1. EMPTY QUERY
    # ========================================================

    if cfg.require_non_empty_query:

        if not query or not query.strip():

            return _build_result(

                decision=
                    EvidenceLevel.INSUFFICIENT,

                reason=
                    "Empty or missing query.",

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

            decision=
                EvidenceLevel.INSUFFICIENT,

            reason=(
                "No relevant evidence was retrieved "
                "from the available knowledge source."
            ),

            confidence=None,

            config=cfg,

            retrieved_chunks=[],

            source_type="unknown",
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

            decision=
                EvidenceLevel.INSUFFICIENT,

            reason=(
                "Insufficient retrieved evidence "
                "to answer safely."
            ),

            confidence=None,

            config=cfg,

            retrieved_chunks=
                retrieved_results,

            source_type=
                _detect_source_type(
                    retrieved_results
                ),
        )


    # ========================================================
    # 5. TOP RESULT
    # ========================================================

    top_chunk = retrieved_results[0]


    # ========================================================
    # 6. EXTRACT SCORES
    # ========================================================

    semantic_score = _safe_float(

        top_chunk.get(
            "semantic_score"
        )
    )

    hybrid_score = _safe_float(

        top_chunk.get(
            "hybrid_score"
        )
    )

    bm25_score = _safe_float(

        top_chunk.get(
            "bm25_score"
        )
    )

    bm25_normalized = _safe_float(

        top_chunk.get(
            "bm25_normalized"
        )
    )


    # ========================================================
    # 7. MISSING SEMANTIC SCORE
    # ========================================================

    if semantic_score is None:

        return _build_result(

            decision=
                EvidenceLevel.INSUFFICIENT,

            reason=(
                "The retrieved result does not contain "
                "a valid semantic_score."
            ),

            confidence=None,

            config=cfg,

            retrieved_chunks=
                retrieved_results,

            source_type=
                _detect_source_type(
                    retrieved_results
                ),
        )


    # ========================================================
    # 8. VALIDATE SEMANTIC SCORE
    # ========================================================

    if not -1.0 <= semantic_score <= 1.0:

        return _build_result(

            decision=
                EvidenceLevel.INSUFFICIENT,

            reason=(
                f"Invalid semantic_score range: "
                f"{semantic_score:.4f}."
            ),

            confidence=None,

            config=cfg,

            retrieved_chunks=
                retrieved_results,

            source_type=
                _detect_source_type(
                    retrieved_results
                ),
        )


    # ========================================================
    # 9. VALIDATE HYBRID SCORE IF AVAILABLE
    # ========================================================

    if hybrid_score is not None:

        if not 0.0 <= hybrid_score <= 1.0:

            hybrid_score = None


    # ========================================================
    # 10. VALIDATE BM25 NORMALIZED SCORE
    # ========================================================

    if bm25_normalized is not None:

        if not 0.0 <= bm25_normalized <= 1.0:

            bm25_normalized = None


    # ========================================================
    # 11. EVIDENCE METRICS
    # ========================================================

    evidence_metrics = {

        "top_semantic_score":
            semantic_score,

        "top_hybrid_score":
            hybrid_score,

        "top_bm25_score":
            bm25_score,

        "top_bm25_normalized":
            bm25_normalized,

        "retrieved_count":
            len(retrieved_results),
    }


    # ========================================================
    # 12. SOURCE TYPE
    # ========================================================

    source_type = _detect_source_type(
        retrieved_results
    )


    # ========================================================
    # 13. SEMANTIC INSUFFICIENT
    # ========================================================

    if (
        semantic_score
        <
        cfg.semantic_insufficient_threshold
    ):

        return _build_result(

            decision=
                EvidenceLevel.INSUFFICIENT,

            reason=(
                f"Top semantic similarity "
                f"({semantic_score:.4f}) is below "
                f"the minimum evidence threshold "
                f"({cfg.semantic_insufficient_threshold:.2f})."
            ),

            confidence=
                semantic_score,

            config=cfg,

            retrieved_chunks=
                retrieved_results,

            source_type=
                source_type,

            evidence_metrics=
                evidence_metrics,
        )


    # ========================================================
    # 14. WEAK EVIDENCE
    # ========================================================
    #
    # Semantic similarity is acceptable but not strong.
    #

    if (
        semantic_score
        <
        cfg.semantic_weak_threshold
    ):

        return _build_result(

            decision=
                EvidenceLevel.WEAK,

            reason=(
                f"Top semantic similarity "
                f"({semantic_score:.4f}) indicates "
                f"weak but usable evidence. "
                f"Generation must remain strictly "
                f"grounded in the retrieved evidence."
            ),

            confidence=
                semantic_score,

            config=cfg,

            retrieved_chunks=
                retrieved_results,

            source_type=
                source_type,

            evidence_metrics=
                evidence_metrics,
        )


    # ========================================================
    # 15. PARTIAL EVIDENCE
    # ========================================================
    #
    # Semantic evidence is reasonably strong.
    # However, it should not automatically be treated
    # as STRONG simply because the top hybrid score is high.
    #

    if (
        semantic_score
        <
        cfg.semantic_strong_threshold
    ):

        return _build_result(

            decision=
                EvidenceLevel.PARTIAL,

            reason=(
                f"Top semantic similarity "
                f"({semantic_score:.4f}) indicates "
                f"partial evidence. "
                f"Only information directly supported "
                f"by the retrieved chunks may be used."
            ),

            confidence=
                semantic_score,

            config=cfg,

            retrieved_chunks=
                retrieved_results,

            source_type=
                source_type,

            evidence_metrics=
                evidence_metrics,
        )


    # ========================================================
    # 16. STRONG EVIDENCE
    # ========================================================
    #
    # Strong semantic similarity.
    #
    # We do NOT require BM25 to be high because a valid
    # semantic question may have weak lexical overlap.
    #
    # However, hybrid/BM25 metrics are preserved for
    # transparency.
    #

    return _build_result(

        decision=
            EvidenceLevel.STRONG,

        reason=(
            f"Top semantic similarity "
            f"({semantic_score:.4f}) meets the "
            f"strong evidence threshold "
            f"({cfg.semantic_strong_threshold:.2f}). "
            f"Answer generation is allowed using only "
            f"the retrieved evidence."
        ),

        confidence=
            semantic_score,

        config=cfg,

        retrieved_chunks=
            retrieved_results,

        source_type=
            source_type,

        evidence_metrics=
            evidence_metrics,
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

        retrieved_results=
            retrieved_results,

        config=config,
    )

    return result["generation_allowed"]


# ============================================================
# REFUSAL MESSAGE
# ============================================================

def get_refusal_message(
    source_type: str = "unknown"
) -> str:

    if source_type == "patient":

        return (
            "I couldn't find enough relevant evidence "
            "in the uploaded patient report to answer "
            "this question reliably."
        )

    if source_type == "core":

        return (
            "I couldn't find enough relevant evidence "
            "in the indexed clinical guidelines to answer "
            "this question reliably."
        )

    if source_type == "mixed":

        return (
            "I couldn't find enough relevant evidence "
            "in the available clinical sources to answer "
            "this question reliably."
        )

    return (
        "I couldn't find enough relevant evidence "
        "in the available sources to answer "
        "this question reliably."
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


    # ========================================================
    # CONFIGURATION
    # ========================================================

    print("\nFinal Retrieval Configuration:")

    print(
        f"Semantic Weight : "
        f"{SEMANTIC_WEIGHT}"
    )

    print(
        f"BM25 Weight     : "
        f"{BM25_WEIGHT}"
    )

    print(
        f"Final Top K     : "
        f"{FINAL_TOP_K}"
    )

    print(
        f"Reranker        : "
        f"{USE_RERANKER}"
    )


    # ========================================================
    # TEST CASES
    # ========================================================
    #
    # NOTE:
    #
    # These semantic scores represent the RAW semantic_score
    # coming from retrieval.py.
    #
    # They are NOT hybrid_score.
    #

    test_scores = [

        0.40,

        0.58,

        0.68,

        0.78,

    ]


    for score in test_scores:

        fake_result = [

            {

                "hybrid_rank":
                    1,

                "chunk_id":
                    "test_chunk",

                "text":
                    "Test retrieved clinical evidence.",

                "metadata": {

                    "document_name":
                        "NICE NG122",

                    "source_type":
                        "core",

                    "section":
                        "Test Section",

                    "page_start":
                        10,

                    "page_end":
                        10,

                    "citation":
                        "[NICE NG122, Test Section, Page 10]",
                },

                # ------------------------------------------------
                # RAW semantic score
                # ------------------------------------------------

                "semantic_score":
                    score,

                # ------------------------------------------------
                # Relative retrieval scores
                # ------------------------------------------------

                "semantic_normalized":
                    1.0,

                "bm25_score":
                    2.0,

                "bm25_normalized":
                    0.5,

                "hybrid_score":
                    0.85,
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
            "RAW SEMANTIC SCORE:",
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
            "SOURCE:",
            result["source_type"]
        )

        print(
            "REASON:",
            result["reason"]
        )


    # ========================================================
    # PATIENT TEST
    # ========================================================

    patient_result = [

        {

            "hybrid_rank":
                1,

            "chunk_id":
                "patient_chunk_001",

            "text":
                "FEV1 was 2.1 L.",

            "metadata": {

                "document_name":
                    "Patient Report",

                "source_type":
                    "patient",

                "section":
                    "Pulmonary Function",

                "page_start":
                    2,

                "page_end":
                    2,

                "citation":
                    "[Patient Report, Pulmonary Function, Page 2]",
            },

            "semantic_score":
                0.79,

            "semantic_normalized":
                1.0,

            "bm25_score":
                1.5,

            "bm25_normalized":
                0.7,

            "hybrid_score":
                0.91,
        }
    ]


    result = check_refusal(

        query=
            "What is my FEV1?",

        retrieved_results=
            patient_result,
    )


    print("\n" + "-" * 75)

    print("PATIENT REPORT TEST")

    print(
        "DECISION:",
        result["decision"]
    )

    print(
        "GENERATION ALLOWED:",
        result["generation_allowed"]
    )

    print(
        "SOURCE:",
        result["source_type"]
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


    # ========================================================
    # EMPTY QUERY
    # ========================================================

    result = check_refusal(

        query="",

        retrieved_results=[],
    )


    print("\n" + "-" * 75)

    print("EMPTY QUERY")

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
    # REFUSAL MESSAGES
    # ========================================================

    print("\n" + "-" * 75)

    print("CORE REFUSAL MESSAGE:")

    print(
        get_refusal_message(
            "core"
        )
    )


    print("\nPATIENT REFUSAL MESSAGE:")

    print(
        get_refusal_message(
            "patient"
        )
    )


    print("\n" + "=" * 75)

    print(
        "EVIDENCE DECISION TEST COMPLETED"
    )

    print("=" * 75)
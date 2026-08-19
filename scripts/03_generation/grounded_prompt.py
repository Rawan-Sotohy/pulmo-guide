"""
============================================================
PULMO GUIDE
DAY 3 - GROUNDED PROMPT BUILDER
============================================================

Pipeline:

    User Query
        ↓
    Hybrid Retrieval
        ↓
    Refusal Gate
        ↓
    THIS MODULE
        ↓
    LLM
        ↓
    Grounded Answer + Citation

This module builds a strict prompt for the LLM.

It does NOT:
- perform retrieval
- modify retrieval configuration
- modify ChromaDB
- call the LLM
- generate the final answer
- create citations itself

It only converts the retrieved evidence into a grounded prompt.
============================================================
"""

from typing import List, Dict, Any


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are Pulmo Guide, a medical information assistant.

Your knowledge source for this answer is ONLY the provided
retrieved evidence from the NICE NG122 knowledge base.

STRICT GROUNDING RULES:

1. Answer ONLY using information explicitly supported by the
   provided evidence.

2. Do NOT use outside medical knowledge, prior knowledge,
   assumptions, or speculation.

3. Do NOT invent facts, recommendations, treatments, numbers,
   or clinical guidance.

4. If the retrieved evidence does not contain enough information
   to answer the question, clearly say that the information is
   not available in the provided NICE NG122 evidence.

5. Every factual medical claim must be traceable to one or more
   provided evidence chunks.

6. Use the provided section and page metadata when citing evidence.

7. Keep the answer concise, clear, and easy to understand.

8. Do not diagnose the user or provide personalized medical
   decisions.

9. If the question asks for information that is outside the
   knowledge base, do not attempt to answer it.

CITATION FORMAT:

Use citations in this format:

[Source: NICE NG122, Section: <section>, Page: <page>]

Place the citation immediately after the claim it supports.

IMPORTANT:
The retrieved evidence is the only source of truth for this answer.
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_grounded_prompt(
    query: str,
    retrieved_results: List[Dict[str, Any]],
) -> str:
    """
    Build a grounded prompt using the user's query and
    retrieved evidence.

    Parameters
    ----------
    query:
        User's question.

    retrieved_results:
        Top retrieved chunks returned by the retrieval pipeline.

    Returns
    -------
    str:
        Complete prompt ready to be sent to an LLM.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not retrieved_results:
        raise ValueError("No retrieved evidence was provided.")

    evidence_blocks = []

    for i, result in enumerate(retrieved_results, start=1):
        metadata = result.get("metadata") or {}

        section = (
            metadata.get("section")
            or metadata.get("section_title")
            or "Unknown section"
        )

        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")

        if page_start is None:
            page_text = "Unknown page"
        elif page_end and page_end != page_start:
            page_text = f"{page_start}-{page_end}"
        else:
            page_text = str(page_start)

        chunk_id = result.get("chunk_id", f"chunk_{i}")

        text = result.get("text", "").strip()

        hybrid_score = result.get("hybrid_score")

        if hybrid_score is not None:
            score_text = f"{float(hybrid_score):.4f}"
        else:
            score_text = "N/A"

        evidence_blocks.append(
            f"""
--- Evidence {i} ---
Chunk ID: {chunk_id}
Hybrid Score: {score_text}
Section: {section}
Page: {page_text}

Text:
{text}
"""
        )

    evidence = "\n".join(evidence_blocks)

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

============================================================
USER QUESTION
============================================================

{query.strip()}

============================================================
RETRIEVED EVIDENCE
============================================================

{evidence}

============================================================
TASK
============================================================

Answer the user's question using ONLY the retrieved evidence.

Before answering, check whether the evidence actually supports
the requested information.

If supported:
- Give a concise answer.
- Include citations immediately after the relevant claims.
- Do not add information that is not present in the evidence.

If not supported:
- Do NOT guess.
- State that the requested information is not available in the
  provided NICE NG122 evidence.

FINAL ANSWER:
"""

    return prompt.strip()


# ============================================================
# SIMPLE MANUAL TEST
# ============================================================

if __name__ == "__main__":

    test_query = (
        "What imaging is offered to people with stage 3 NSCLC?"
    )

    test_results = [
        {
            "hybrid_rank": 1,
            "chunk_id": "chunk_0040",
            "text": (
                "Offer contrast-enhanced brain MRI for people "
                "with stage 3 NSCLC."
            ),
            "metadata": {
                "section": "Further staging",
                "page_start": 14,
                "page_end": 14,
            },
            "hybrid_score": 0.91,
        },
        {
            "hybrid_rank": 2,
            "chunk_id": "chunk_0038",
            "text": (
                "Confirm the presence of isolated distant "
                "metastases before treatment."
            ),
            "metadata": {
                "section": "Further staging",
                "page_start": 14,
                "page_end": 14,
            },
            "hybrid_score": 0.87,
        },
    ]

    print("=" * 70)
    print("PULMO GUIDE - GROUNDED PROMPT BUILDER TEST")
    print("=" * 70)

    prompt = build_grounded_prompt(
        query=test_query,
        retrieved_results=test_results,
    )

    print("\nGenerated Prompt:\n")
    print(prompt)

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)
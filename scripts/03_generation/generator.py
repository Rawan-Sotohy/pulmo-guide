"""
============================================================
PULMO GUIDE
LLM GENERATOR
============================================================

FINAL PIPELINE:

    Safety Decision
          |
          v
    Hybrid Retrieval
    Semantic 70% + BM25 30%
          |
          v
    Final Top 5
          |
          v
    Evidence Decision
          |
          v
    Citation Builder
          |
          v
    Grounded Prompt
          |
          v
    THIS MODULE
          |
          v
    Gemini LLM
          |
          v
    Raw Grounded Answer

IMPORTANT:

- NO reranker is used.
- Retrieval is already finalized before this module.
- Citations are created by citation.py.
- The LLM MUST NOT create or modify citations.
- This module does NOT perform retrieval.
- This module does NOT perform safety checks.
- This module does NOT perform evidence scoring.
- This module does NOT create citations.
- This module does NOT modify ChromaDB.

It only sends the already-approved grounded prompt
to the LLM and returns the raw generated answer.
============================================================
"""

import os
from typing import Dict, Any

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

GENERATION_MODEL = os.getenv(
    "PULMO_GENERATION_MODEL",
    "gemini-3.6-flash",
)

GENERATION_TEMPERATURE = 0.0


# ============================================================
# GEMINI CLIENT
# ============================================================

def _get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Make sure it exists in the project .env file."
        )

    try:

        from google import genai

    except ImportError:

        raise RuntimeError(
            "google-genai is not installed. "
            "Run: pip install google-genai"
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    grounded_prompt: str,
) -> Dict[str, Any]:
    """
    Generate an answer using the already-built grounded prompt.

    This function ONLY performs the LLM generation step.

    The grounded prompt already contains:

    - safety/persona policy
    - evidence level
    - retrieved evidence
    - verified citations
    - grounding rules
    - final answer instructions

    The LLM is NOT responsible for creating citations.
    """

    if not grounded_prompt or not grounded_prompt.strip():

        raise ValueError(
            "Grounded prompt cannot be empty."
        )

    client = _get_gemini_client()

    try:

        from google.genai import types

        response = client.models.generate_content(

            model=GENERATION_MODEL,

            contents=grounded_prompt,

            config=types.GenerateContentConfig(
                temperature=GENERATION_TEMPERATURE,
            ),
        )

    except Exception as e:

        raise RuntimeError(
            f"Gemini generation failed: {e}"
        ) from e

    answer = getattr(
        response,
        "text",
        None
    )

    if not answer or not answer.strip():

        raise RuntimeError(
            "LLM returned an empty response."
        )

    answer = answer.strip()

    return {

        "status": "generated",

        "model": GENERATION_MODEL,

        "temperature": GENERATION_TEMPERATURE,

        "answer": answer,

        "raw_answer": answer,
    }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("PULMO GUIDE - DAY 3 LLM GENERATOR TEST")
    print("=" * 75)

    print("\nMODEL:")
    print(GENERATION_MODEL)

    print("\nTEMPERATURE:")
    print(GENERATION_TEMPERATURE)

    test_prompt = """
You are Pulmo Guide, a safety-focused medical information assistant.

Use ONLY the retrieved evidence below.

Do NOT use outside medical knowledge.
Do NOT invent facts.

IMPORTANT CITATION RULE:

The citation below has already been verified by the system.

You MUST:
- use the provided citation exactly as written
- NOT create a new citation
- NOT change the document name
- NOT change the section
- NOT change the page
- NOT invent a citation

LANGUAGE POLICY:

Answer in the same language as the user's question.

The evidence may remain in its original language.

USER QUESTION:

What imaging is offered to people with stage 3 NSCLC?

RETRIEVED EVIDENCE:

Document: NICE NG122

Section: Further staging

Page: 14

Text:
Offer contrast-enhanced brain MRI for people with stage 3 NSCLC.

VERIFIED CITATION:
[NICE NG122, Further staging, Page 14]

FINAL ANSWER:

Answer:
"""

    try:

        result = generate_answer(
            grounded_prompt=test_prompt
        )

        print("\nSTATUS:")
        print(result["status"])

        print("\nMODEL:")
        print(result["model"])

        print("\nTEMPERATURE:")
        print(result["temperature"])

        print("\nGENERATED ANSWER:")
        print(result["answer"])

        print("\n" + "=" * 75)
        print("GENERATOR TEST COMPLETED")
        print("=" * 75)

    except Exception as e:

        print("\nGENERATION ERROR:")
        print(str(e))
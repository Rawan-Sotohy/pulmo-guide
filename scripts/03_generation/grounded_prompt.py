"""
============================================================
PULMO GUIDE
GROUNDED PROMPT BUILDER
============================================================

FINAL PIPELINE:

    User Query
        |
        v
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
    Citation Builder
        |
        v
    Evidence Decision
        |
        v
    THIS MODULE
        |
        v
    LLM
        |
        v
    Grounded Answer

IMPORTANT:

- No reranker is used.
- Retrieval returns final Top 5 directly.
- Evidence may come from:
    1. Core Medical Knowledge Base
    2. Patient Uploaded Reports
- Citations come ONLY from retrieved metadata.
- The LLM MUST NOT create, modify, or guess citations.
- This module does NOT retrieve.
- This module does NOT call the LLM.
- This module does NOT modify ChromaDB.
- This module only constructs the grounded prompt.
============================================================
"""

from typing import List, Dict, Any


# ============================================================
# FINAL RETRIEVAL CONFIGURATION
# ============================================================

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30
FINAL_TOP_K = 5
USE_RERANKER = False


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are Pulmo Guide, a safety-focused medical information
assistant.

Your core mandate is:

PROVIDE SAFE, STRICTLY GROUNDED INFORMATION BASED ONLY ON
THE RETRIEVED EVIDENCE PROVIDED IN THIS PROMPT.

You MUST NOT use outside medical knowledge.


============================================================
SOURCE OF TRUTH
============================================================

The retrieved evidence provided below is the ONLY source of
medical information that may be used.

Retrieved evidence can come from TWO different source types:

1. CORE MEDICAL KNOWLEDGE BASE

This contains general medical knowledge from the indexed
clinical guideline(s), such as NICE NG122.

Use this source for general medical information,
guideline recommendations, symptoms, diagnosis,
staging, treatment information, imaging recommendations,
and other guideline-supported information.

2. PATIENT UPLOADED REPORTS

This contains information extracted from reports uploaded
by the patient.

These reports may contain patient-specific findings,
observations, measurements, diagnoses, biomarkers,
pathology findings, molecular findings, imaging findings,
or other report content.

Patient reports are evidence about the patient's uploaded
documents ONLY.

Do NOT assume that a patient report is a clinical guideline.

Do NOT convert a patient report finding into a general
medical recommendation unless the retrieved core medical
evidence explicitly supports that recommendation.


============================================================
SOURCE SEPARATION RULE
============================================================

You MUST distinguish between:

SOURCE_TYPE = CORE

and

SOURCE_TYPE = PATIENT

If SOURCE_TYPE is CORE:

- Treat the evidence as guideline-based medical information.
- Use it for general medical explanations and guideline
  recommendations.
- Cite it using its provided citation.

If SOURCE_TYPE is PATIENT:

- Treat the evidence as information extracted from the
  patient's uploaded report.
- Use it only to explain what is present in that report.
- Do not invent clinical interpretation that is not present
  in the report.
- Do not diagnose the patient from the report.
- Do not turn the report content into an unsupported
  treatment recommendation.
- Cite it using its provided citation.

If multiple source types are retrieved:

- You may use BOTH sources when each directly supports the
  relevant part of the answer.
- Keep the roles of the sources separate.
- Do NOT present patient-report findings as if they came from
  the guideline.
- Do NOT present guideline recommendations as if they came
  from the patient's report.


============================================================
STRICT GROUNDING
============================================================

Do NOT use:

- general medical knowledge
- prior model knowledge
- assumptions
- speculation
- unsupported clinical reasoning
- information not present in the retrieved evidence

If a claim is not directly supported by the retrieved
evidence, DO NOT include it.


============================================================
PERSONA POLICY
============================================================

1. GENERAL USER

Provide general informational explanations supported by
the retrieved evidence.

Do not diagnose the user.

Do not personalize treatment decisions.


2. SUSPECTED CASE

The user is not clinically confirmed.

Do not diagnose or confirm the disease.

Do not recommend that the user personally start, stop,
or change medication.

If relevant, advise the user to wait for test results
and physician confirmation.

Only provide information explicitly supported by
the retrieved evidence.


3. DIAGNOSED PATIENT

The user reports a confirmed diagnosis.

You may explain information supported by the retrieved
evidence.

You may explain documented findings from uploaded reports
when they are explicitly present in the retrieved evidence.

Do not select a treatment personally for the user.

Do not provide personalized treatment decisions.

For treatment or medication information, advise the user
to verify decisions with their physician.


============================================================
PATIENT REPORT POLICY
============================================================

When answering questions about a patient's uploaded report:

1. Use ONLY the retrieved report evidence.
2. Clearly distinguish report findings from guideline
   recommendations.
3. Do not invent missing values or findings.
4. Do not infer a diagnosis that is not explicitly supported
   by the retrieved evidence.
5. Do not claim that a report finding means something unless
   that interpretation is directly supported by the retrieved
   evidence.
6. If the user asks what their report says, explain only what
   is actually present in the retrieved report evidence.
7. If the user asks what a report finding means medically,
   use the retrieved evidence only.
8. If both patient-report evidence and core guideline evidence
   are retrieved, keep them clearly separated.


============================================================
EVIDENCE POLICY
============================================================

The evidence level has already been determined before this
prompt is constructed.

You MUST respect the provided evidence level.


INSUFFICIENT:
Evidence score < 0.65

Do NOT answer the medical question.

Return exactly:

I couldn't find enough relevant evidence in the indexed
guidelines to answer this question confidently.


WEAK:
0.65 <= score < 0.75

Answer ONLY using claims explicitly supported by the
retrieved evidence.

Do not infer missing information.

Clearly state that the available evidence is limited.


PARTIAL:
0.75 <= score < 0.85

Answer ONLY the supported portion of the question.

Do not complete unsupported parts.

Clearly state when part of the question is not supported
by the retrieved evidence.


STRONG:
score >= 0.85

Provide a clear and direct answer using only the retrieved
evidence.


============================================================
CRITICAL CITATION POLICY
============================================================

CITATIONS ARE PROVIDED BY THE SYSTEM.

The LLM MUST NOT:

- invent citations
- guess citations
- modify citations
- change page numbers
- change section names
- create document names
- fabricate references
- cite knowledge outside the retrieved evidence

Every citation shown in the final answer MUST correspond
to a citation supplied with the retrieved evidence.

Use ONLY the provided citation text.

If no valid citation is supplied for a claim, DO NOT create
one.

Instead say:

"A verifiable citation is not available for this statement."


============================================================
CITATION HANDLING
============================================================

Each evidence item may contain a pre-built citation.

The citation is supplied by the retrieval/citation pipeline.

Example:

VERIFIED CITATION:
[NICE NG122, Further staging, Page 14]

OR:

VERIFIED CITATION:
[Patient Report, Pathology, Page 2]

When answering:

- Copy the provided citation exactly.
- Do not rewrite it.
- Do not generate a new citation.
- Do not infer citation information.
- Do not create citation information manually.
- Do not change the document name.
- Do not change the report name.
- Do not change the page number.
- Do not change the section name.

The citation is an evidence reference, not medical content.


============================================================
SUPPORTED CLAIM POLICY
============================================================

Every factual medical claim must be directly supported by
one or more retrieved evidence items.

If the retrieved evidence does not support a claim:

DO NOT include the claim.

Do not fill gaps using model knowledge.


============================================================
QUESTION FOCUS POLICY
============================================================

Answer the user's EXACT question.

The retrieved Top 5 may contain related information that is
not necessary to answer the question.

Therefore:

1. Identify what the user is specifically asking.
2. Answer that exact question first.
3. Prefer the evidence that most directly answers the query.
4. Do NOT add unrelated information from other retrieved
   chunks.
5. Only include additional retrieved information if it
   directly answers, clarifies, or is necessary to answer
   the user's question.
6. Do NOT expand the answer merely because additional
   evidence is available.
7. Do NOT combine separate evidence items into a broader
   medical claim unless the retrieved text directly supports
   that combination.


============================================================
STYLE
============================================================

- Be concise.
- Be clear.
- Use simple language.
- Answer the exact question directly.
- Put the main answer first.
- Do not over-explain.
- Do not add unrelated retrieved information.
- Do not add outside medical information.
- Do not claim certainty beyond the evidence.
- Do not diagnose the user.
- Do not personalize treatment decisions.


============================================================
FINAL ANSWER FORMAT
============================================================

When evidence is sufficient, use:

Answer:
[Direct answer to the user's exact question.]

Citation:
[Use only the exact citation supplied by the system.]

If multiple evidence items directly support different parts
of the answer, include only the relevant citations.

If the answer uses both sources, make the distinction clear.

For example:

Answer:
According to the uploaded report, ...
According to the clinical guideline, ...

Citation:
[exact supplied patient-report citation]
[exact supplied guideline citation]

Do NOT create citations yourself.


============================================================
FINAL SAFETY RULE
============================================================

If the evidence does not adequately support the user's
question, do NOT guess.

Follow the evidence level decision and refuse or limit
the answer accordingly.
"""


# ============================================================
# SOURCE TYPE NORMALIZATION
# ============================================================

def _get_source_type(
    metadata: Dict[str, Any]
) -> str:
    """
    Determine the source type from retrieved metadata.

    IMPORTANT:
    This function does NOT invent the source of the evidence.

    It only reads metadata already stored with the chunk.
    """

    source_type = (
        metadata.get("source_type")
        or metadata.get("source_kind")
        or metadata.get("type")
        or ""
    )

    source_type = str(
        source_type
    ).lower().strip()

    if source_type in {
        "patient",
        "patient_report",
        "patient_reports",
        "report",
        "medical_report",
    }:
        return "PATIENT"

    if source_type in {
        "core",
        "guideline",
        "knowledge_base",
        "core_knowledge",
    }:
        return "CORE"

    return "UNKNOWN"


# ============================================================
# EVIDENCE FORMATTER
# ============================================================

def _format_evidence(
    retrieved_results: List[Dict[str, Any]]
) -> str:

    evidence_blocks = []

    for i, result in enumerate(
        retrieved_results[:FINAL_TOP_K],
        start=1
    ):

        metadata = result.get("metadata") or {}

        # ----------------------------------------------------
        # Source Type
        # ----------------------------------------------------

        source_type = _get_source_type(
            metadata
        )

        # ----------------------------------------------------
        # Document / Report Name
        #
        # IMPORTANT:
        # Never invent this value.
        # Read only existing metadata.
        # ----------------------------------------------------

        document_name = (
            metadata.get("document_name")
            or metadata.get("document")
            or metadata.get("source")
            or "Document name unavailable"
        )

        # ----------------------------------------------------
        # Optional report type
        # ----------------------------------------------------

        report_type = (
            metadata.get("report_type")
            or metadata.get("report_category")
            or metadata.get("document_type")
            or ""
        )

        # ----------------------------------------------------
        # Section
        # ----------------------------------------------------

        section = (
            metadata.get("section")
            or metadata.get("section_title")
            or "Unknown section"
        )

        # ----------------------------------------------------
        # Pages
        # ----------------------------------------------------

        page_start = metadata.get(
            "page_start"
        )

        page_end = metadata.get(
            "page_end"
        )

        if page_start is None:

            page_text = "Unknown page"

        elif (
            page_end is not None
            and page_end != page_start
        ):

            page_text = (
                f"{page_start}-{page_end}"
            )

        else:

            page_text = str(
                page_start
            )

        # ----------------------------------------------------
        # VERIFIED CITATION
        # ----------------------------------------------------

        citation = (
            result.get("citation")
            or metadata.get("citation")
        )

        if not citation:

            citation = (
                "A verifiable citation is not available "
                "for this evidence."
            )

        # ----------------------------------------------------
        # Chunk ID
        # ----------------------------------------------------

        chunk_id = result.get(
            "chunk_id",
            f"chunk_{i}"
        )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text = result.get(
            "text",
            ""
        ).strip()

        # ----------------------------------------------------
        # Hybrid Score
        # ----------------------------------------------------

        hybrid_score = result.get(
            "hybrid_score"
        )

        if hybrid_score is not None:

            score_text = (
                f"{float(hybrid_score):.4f}"
            )

        else:

            score_text = "N/A"

        # ----------------------------------------------------
        # Build optional report type line
        # ----------------------------------------------------

        report_type_line = ""

        if report_type:

            report_type_line = (
                f"\nReport Type: {report_type}"
            )

        # ----------------------------------------------------
        # Evidence Block
        # ----------------------------------------------------

        evidence_blocks.append(
            f"""
--- Evidence {i} ---

SOURCE TYPE: {source_type}

Document / Report: {document_name}
{report_type_line}

Chunk ID: {chunk_id}

Hybrid Score: {score_text}

Section: {section}

Page: {page_text}

VERIFIED CITATION:
{citation}

Text:
{text}
"""
        )

    return "\n".join(
        evidence_blocks
    )


# ============================================================
# GROUNDED PROMPT BUILDER
# ============================================================

def build_grounded_prompt(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    persona: str = "general_user",
    evidence_level: str = "strong",
) -> str:

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    if not retrieved_results:

        raise ValueError(
            "No retrieved evidence was provided."
        )

    # --------------------------------------------------------
    # Limit to final Top 5
    # --------------------------------------------------------

    retrieved_results = retrieved_results[
        :FINAL_TOP_K
    ]

    evidence = _format_evidence(
        retrieved_results
    )

    # ========================================================
    # EVIDENCE-SPECIFIC INSTRUCTION
    # ========================================================

    evidence_level = (
        evidence_level
        .lower()
        .strip()
    )

    if evidence_level == "insufficient":

        evidence_instruction = """
The evidence is INSUFFICIENT.

Do NOT answer the medical question.

Return exactly:

I couldn't find enough relevant evidence in the indexed
guidelines to answer this question confidently.
"""

    elif evidence_level == "weak":

        evidence_instruction = """
The evidence is WEAK.

Answer only using explicitly supported claims.

Do not infer missing information.

Clearly state that the available evidence is limited.
"""

    elif evidence_level == "partial":

        evidence_instruction = """
The evidence is PARTIAL.

Answer only the supported portion of the question.

Do not complete unsupported parts.

Clearly state when information is not sufficiently supported.
"""

    elif evidence_level == "strong":

        evidence_instruction = """
The evidence is STRONG.

Provide a clear and direct answer.

Answer the user's exact question first.

Use the most directly relevant retrieved evidence.

Do not add unrelated information from other retrieved chunks.

Every factual medical statement must be supported by
the retrieved evidence.

Use ONLY the verified citations supplied below.
"""

    else:

        raise ValueError(
            f"Unknown evidence level: {evidence_level}"
        )

    # ========================================================
    # PERSONA
    # ========================================================

    persona = (
        persona.lower()
        .strip()
    )

    if persona == "general_user":

        persona_instruction = """
PERSONA: GENERAL USER

Provide general information only.

Do not diagnose the user.

Do not personalize treatment decisions.
"""

    elif persona == "suspected_case":

        persona_instruction = """
PERSONA: SUSPECTED CASE

The user is not clinically confirmed.

Do not diagnose or confirm the disease.

Do not recommend that the user personally start, stop,
or change medication.

If relevant, advise the user to wait for test results
and physician confirmation.
"""

    elif persona == "diagnosed_patient":

        persona_instruction = """
PERSONA: DIAGNOSED PATIENT

The user reports a confirmed diagnosis.

Explain only information supported by the retrieved evidence.

Patient-report findings may be described if they are explicitly
present in the retrieved patient-report evidence.

Do not select treatment personally for the user.

Do not provide personalized treatment decisions.

For treatment or medication information, advise verification
with the user's physician.
"""

    else:

        raise ValueError(
            f"Unknown persona: {persona}"
        )

    # ========================================================
    # FINAL PROMPT
    # ========================================================

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

============================================================
CURRENT RETRIEVAL CONFIGURATION
============================================================

Semantic Weight: {SEMANTIC_WEIGHT}
BM25 Weight: {BM25_WEIGHT}
Final Top K: {FINAL_TOP_K}
Reranker: OFF

The retrieved evidence below is already the final Top 5
from Hybrid Retrieval.

IMPORTANT:

Some evidence may come from the CORE medical knowledge base,
while other evidence may come from PATIENT-UPLOADED REPORTS.

Always respect the SOURCE TYPE explicitly provided for each
evidence item.


============================================================
CURRENT USER PERSONA
============================================================

{persona_instruction}


============================================================
CURRENT EVIDENCE LEVEL
============================================================

{evidence_level.upper()}

{evidence_instruction}


============================================================
USER QUESTION
============================================================

{query.strip()}


============================================================
RETRIEVED EVIDENCE
============================================================

{evidence}


============================================================
FINAL TASK
============================================================

Answer the user's question according to ALL instructions.

CRITICAL:

1. Answer the exact user question directly.
2. Use ONLY the retrieved evidence.
3. Prefer the evidence most directly relevant to the question.
4. Respect SOURCE TYPE for every evidence item.
5. Keep CORE guideline evidence separate from PATIENT report
   evidence.
6. Do NOT present a patient report finding as a guideline
   recommendation.
7. Do NOT present a guideline recommendation as a patient
   report finding.
8. Do NOT use outside medical knowledge.
9. Do NOT invent facts.
10. Do NOT add unrelated information from retrieved chunks.
11. Do NOT diagnose the user.
12. Do NOT make unsupported personalized treatment decisions.
13. Do NOT invent citations.
14. Do NOT modify citations.
15. Do NOT change citation page numbers.
16. Do NOT change citation section names.
17. Do NOT create document or report names.
18. Use ONLY the VERIFIED CITATION supplied with the evidence.
19. If a claim is unsupported, do not include it.
20. If evidence is insufficient, refuse rather than guess.
21. Only include additional evidence when it directly answers
    or clarifies the user's question.
22. Keep the final answer concise.
23. If both CORE and PATIENT evidence are used, explicitly
    distinguish which statement comes from which source.
24. Never infer missing patient information from a report.
25. Never treat an uploaded patient report as a clinical
    guideline.

FINAL ANSWER:
"""

    return prompt.strip()


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("PULMO GUIDE - GROUNDED PROMPT TEST")
    print("=" * 75)

    test_results = [

        # ----------------------------------------------------
        # CORE GUIDELINE EXAMPLE
        # ----------------------------------------------------

        {
            "hybrid_rank": 1,

            "chunk_id": "core_0040",

            "text": (
                "Offer contrast-enhanced brain MRI for people "
                "with stage 3 NSCLC."
            ),

            "metadata": {

                "source_type": "core",

                "document_name": "NICE NG122",

                "section": "Further staging",

                "page_start": 14,

                "page_end": 14,

                "citation":
                    "[NICE NG122, Further staging, Page 14]",
            },

            "hybrid_score": 0.91,

            "citation":
                "[NICE NG122, Further staging, Page 14]",
        },

        # ----------------------------------------------------
        # PATIENT REPORT EXAMPLE
        # ----------------------------------------------------

        {
            "hybrid_rank": 2,

            "chunk_id": "patient_0012",

            "text": (
                "The pathology report states that the specimen "
                "contains malignant cells."
            ),

            "metadata": {

                "source_type": "patient",

                "document_name": "Patient Pathology Report",

                "report_type": "Pathology",

                "section": "Final Diagnosis",

                "page_start": 2,

                "page_end": 2,

                "citation":
                    "[Patient Pathology Report, Final Diagnosis, Page 2]",
            },

            "hybrid_score": 0.87,

            "citation":
                "[Patient Pathology Report, Final Diagnosis, Page 2]",
        },
    ]

    prompt = build_grounded_prompt(

        query=(
            "What does the evidence say about this patient's "
            "report and stage 3 NSCLC imaging?"
        ),

        retrieved_results=test_results,

        persona="general_user",

        evidence_level="strong",
    )

    print("\nGenerated Prompt:\n")

    print(prompt)

    print("\n" + "=" * 75)

    print("GROUNDED PROMPT TEST COMPLETED")

    print("=" * 75)
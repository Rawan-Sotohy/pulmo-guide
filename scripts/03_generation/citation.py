"""
============================================================
PULMO GUIDE
CITATION BUILDER
============================================================

This module:
- DOES NOT call the LLM
- DOES NOT perform retrieval
- DOES NOT generate medical content
- DOES NOT invent citation information

It builds citations ONLY from retrieved chunk metadata.

Format:

[Document Name, Section, Page N]
============================================================
"""

from typing import Dict, Any, List


# ============================================================
# SINGLE CITATION
# ============================================================

def build_citation(
    metadata: Dict[str, Any]
) -> str:
    """
    Build a citation from retrieved chunk metadata.

    Citation information must come from the metadata.
    Nothing is invented by the LLM.
    """

    if not metadata:
        return "[Citation unavailable]"

    document = (
        metadata.get("document_name")
        or metadata.get("document")
    )

    if not document:

        source_type = metadata.get("source_type")

        if source_type == "core":
             document = "NICE NG122"

        else:
             document = "Unknown Document"

    section = (
        metadata.get("section")
        or metadata.get("section_title")
    )

    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")

    # --------------------------------------------------------
    # Document
    # --------------------------------------------------------

    if not document:
        document = "Unknown Document"

    # --------------------------------------------------------
    # Section
    # --------------------------------------------------------

    if not section:
        section = "Unknown Section"

    # --------------------------------------------------------
    # Page
    # --------------------------------------------------------

    if page_start is None:
        page_text = "Unknown Page"

    elif (
        page_end is not None
        and page_end != page_start
    ):
        page_text = f"{page_start}-{page_end}"

    else:
        page_text = str(page_start)

    return (
        f"[{document}, "
        f"{section}, "
        f"Page {page_text}]"
    )


# ============================================================
# MULTIPLE CITATIONS
# ============================================================

def build_citations(
    retrieved_results: List[Dict[str, Any]]
) -> List[str]:
    """
    Build citations for multiple retrieved results.

    Only metadata from retrieved results is used.
    """

    citations = []

    for result in retrieved_results:

        metadata = result.get("metadata") or {}

        citation = build_citation(
            metadata
        )

        if citation not in citations:
            citations.append(citation)

    return citations


# ============================================================
# FORMAT CITATIONS FOR FINAL ANSWER
# ============================================================

def format_citations(
    retrieved_results: List[Dict[str, Any]]
) -> str:
    """
    Return citations as a clean multiline block.
    """

    citations = build_citations(
        retrieved_results
    )

    if not citations:
        return "[Citation unavailable]"

    return "\n".join(citations)


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("PULMO GUIDE - DAY 3 CITATION BUILDER TEST")
    print("=" * 75)

    test_results = [

        {
            "chunk_id": "chunk_0040",

            "text": (
                "Offer contrast-enhanced brain MRI "
                "for people with stage 3 NSCLC."
            ),

            "metadata": {
                "document_name": "NICE NG122",
                "section": "Further staging",
                "page_start": 14,
                "page_end": 14,
            },
        },

        {
            "chunk_id": "chunk_0038",

            "text": (
                "Confirm the presence of isolated "
                "distant metastases before treatment."
            ),

            "metadata": {
                "document_name": "NICE NG122",
                "section": "Further staging",
                "page_start": 14,
                "page_end": 14,
            },
        },
    ]

    # --------------------------------------------------------
    # Test 1 - Single citation
    # --------------------------------------------------------

    print("\nSingle Citation:")

    citation = build_citation(
        test_results[0]["metadata"]
    )

    print(citation)

    # --------------------------------------------------------
    # Test 2 - Multiple citations
    # --------------------------------------------------------

    print("\nMultiple Citations:")

    citations = build_citations(
        test_results
    )

    for citation in citations:
        print(citation)

    # --------------------------------------------------------
    # Test 3 - Formatted citations
    # --------------------------------------------------------

    print("\nFormatted Citation Block:")

    print(
        format_citations(
            test_results
        )
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("CITATION TEST COMPLETED")
    print("=" * 75)
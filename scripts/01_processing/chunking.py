from pathlib import Path
import json
import re
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# Embedding model
# Used ONLY for semantic grouping during chunking.
# The same model can be used later for retrieval.
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ============================================================
# Chunk configuration
# ============================================================

CONFIG = {

    "core": {
        "max_words": 700,
        "overlap_words": 100,

        # This value is experimental.
        # Tune later using retrieval evaluation.
        "semantic_threshold": 0.55,
    },

    "patient": {
        "max_words": 350,
        "overlap_words": 50,

        # This value is experimental.
        # Tune later using retrieval evaluation.
        "semantic_threshold": 0.50,
    }
}


# ============================================================
# Helpers
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text):
    """
    Split normal prose into sentences.

    Headings are handled separately and are NOT passed here.
    """

    text = normalize_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def word_count(text):
    return len(text.split())


def cosine_similarity(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def unique_preserve_order(items):

    result = []

    for item in items:

        if item not in result:
            result.append(item)

    return result


# ============================================================
# Heading helpers
# ============================================================

def is_numbering_only(text):
    """
    Detect recommendation numbers such as:

        1.4.6
        1.2.19
        1.6.10

    These are NOT useful as standalone chunks.
    """

    text = normalize_text(text)

    return bool(
        re.fullmatch(
            r"\d+(?:\.\d+)+",
            text
        )
    )


def is_heading_element(element):
    """
    Decide whether an element is a heading.

    Primary signal:
        is_section_heading

    Secondary signal:
        very short numbering-only elements are NOT treated
        as useful headings.

    Fallback signal:
        Some parsers fail to set is_section_heading on
        front-matter headings (for example "Your responsibility"),
        even though they use the exact same visual heading style
        (font family, size, bold flag) as headings elsewhere in
        the same document that ARE correctly tagged.

        We only trust this fallback for style combinations that
        are confirmed, in this document, to belong exclusively to
        real section headings:

            Lora-SemiBold at 16.5pt or larger
            Inter-SemiBold at 13.5pt or larger

        Inter-SemiBold at 12pt is deliberately excluded: it is
        used for inline sub-headings inside recommendation
        rationale text (e.g. "Why the committee made the
        recommendations"), which are NOT standalone section
        headings and must stay classified as normal content.
    """

    text = normalize_text(
        element.get("text", "")
    )

    if not text:
        return False

    if is_numbering_only(text):
        return False

    if element.get(
        "is_section_heading",
        False
    ):
        return True

    font = element.get("font", "") or ""
    size = element.get("size", 0) or 0
    flags = element.get("flags", 0)

    if flags == 20:

        if (
            font.startswith("Lora-SemiBold")
            and size >= 16.5
        ):
            return True

        if (
            font.startswith("Inter-SemiBold")
            and size >= 13.5
        ):
            return True

    return False


# ============================================================
# Citation / Metadata
# ============================================================

def build_source_metadata(elements):
    """
    Preserve everything needed to trace a chunk
    back to the original medical document.
    """

    pages = []
    documents = []
    source_types = []
    element_ids = []
    bboxes = []

    for element in elements:

        page = element.get("page")

        if page is not None:
            pages.append(page)

        document = element.get("document")

        if document:
            documents.append(document)

        source_type = element.get("source_type")

        if source_type:
            source_types.append(source_type)

        element_id = element.get("id")

        if element_id:
            element_ids.append(element_id)

        bbox = element.get("bbox")

        if bbox:
            bboxes.append(bbox)

    pages = sorted(set(pages))

    documents = unique_preserve_order(
        documents
    )

    source_types = unique_preserve_order(
        source_types
    )

    if pages:

        if len(pages) == 1:

            page_text = (
                f"p. {pages[0]}"
            )

        else:

            page_text = (
                "pp. "
                + ", ".join(
                    str(p)
                    for p in pages
                )
            )

    else:

        page_text = "page unavailable"

    document_name = (
        documents[0]
        if documents
        else "Unknown document"
    )

    citation = (
        f"{document_name}, {page_text}"
    )

    return {

        "pages": pages,

        "documents": documents,

        "source_types": source_types,

        "element_ids": element_ids,

        "bboxes": bboxes,

        "citation": citation,
    }


# ============================================================
# Build semantic units
# ============================================================

def build_units(elements):
    """
    Convert parser elements into semantic units.

    IMPORTANT HEADING LOGIC:

    A heading is kept as a separate unit internally,
    but it is NEVER allowed to become a standalone final chunk.

    Later, heading + following content are attached together.
    """

    units = []

    for element in elements:

        text = normalize_text(
            element.get("text", "")
        )

        if not text:
            continue

        heading = is_heading_element(
            element
        )

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        if heading:

            units.append({
                "text": text,
                "element": element,
                "is_heading": True,
            })

            continue

        # ----------------------------------------------------
        # Numbering-only elements
        #
        # Example:
        # 1.4.6
        #
        # Do not create a useless chunk for them.
        # ----------------------------------------------------

        if is_numbering_only(text):

            units.append({
                "text": text,
                "element": element,
                "is_heading": False,
                "is_numbering": True,
            })

            continue

        # ----------------------------------------------------
        # Normal content
        # ----------------------------------------------------

        sentences = split_sentences(text)

        if not sentences:
            continue

        for sentence in sentences:

            units.append({
                "text": sentence,
                "element": element,
                "is_heading": False,
                "is_numbering": False,
            })

    return units


# ============================================================
# Attach headings to following content
# ============================================================

def attach_headings_to_content(units):
    """
    IMPORTANT:

    A heading should provide context to the content
    that follows it.

    Example:

        Heading
        Sentence A
        Sentence B

    becomes one logical group:

        Heading + Sentence A + Sentence B

    A heading is never returned as an independent group.

    FIX:

    A heading must start a NEW logical group. Without closing
    off the current_group when a new heading is seen, every
    heading in a section gets bundled into one giant group,
    and headings end up detached from the content that
    immediately follows them.
    """

    if not units:
        return []

    groups = []

    current_group = []

    pending_heading = None

    for unit in units:

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        if unit["is_heading"]:

            # A heading starts a new logical group — close off
            # whatever content we've collected so far so it
            # doesn't get bundled with the next heading's
            # content.
            if current_group:

                groups.append(
                    current_group
                )

                current_group = []

            # If a previous heading has no content,
            # keep the latest heading.
            pending_heading = unit

            continue

        # ----------------------------------------------------
        # Numbering-only
        # ----------------------------------------------------

        if unit.get("is_numbering", False):

            # Keep numbering attached to following content
            # instead of allowing it to become a standalone chunk.

            if pending_heading is not None:

                current_group.append(
                    pending_heading
                )

                pending_heading = None

            current_group.append(unit)

            continue

        # ----------------------------------------------------
        # Normal content
        # ----------------------------------------------------

        if pending_heading is not None:

            # Heading becomes part of the content group.
            current_group.append(
                pending_heading
            )

            pending_heading = None

        current_group.append(unit)

    # --------------------------------------------------------
    # Remaining heading with no content
    #
    # We DO NOT create a standalone heading chunk.
    # --------------------------------------------------------

    if current_group:

        groups.append(
            current_group
        )

    return groups


# ============================================================
# Semantic grouping
# ============================================================

def semantic_group_units(
    units,
    model,
    threshold
):
    """
    Group consecutive semantic units.

    Rules:

    1. Never cross a heading boundary.
    2. Never compare distant units.
    3. Semantic similarity only decides whether
       consecutive content should stay together.
    4. A numbering marker (e.g. "1.4.6") must always stay
       glued to the content that follows it, regardless of
       semantic similarity — otherwise it can end up alone
       in its own chunk.
    """

    if not units:
        return []

    # --------------------------------------------------------
    # First attach headings to their following content.
    # --------------------------------------------------------

    heading_groups = (
        attach_headings_to_content(units)
    )

    if not heading_groups:
        return []

    final_groups = []

    for logical_group in heading_groups:

        if not logical_group:
            continue

        # ----------------------------------------------------
        # If the group only contains a heading,
        # discard it.
        # ----------------------------------------------------

        has_content = any(
            not unit["is_heading"]
            for unit in logical_group
        )

        if not has_content:
            continue

        # ----------------------------------------------------
        # Heading stays with its following content.
        # ----------------------------------------------------

        heading_units = [
            unit
            for unit in logical_group
            if unit["is_heading"]
        ]

        content_units = [
            unit
            for unit in logical_group
            if not unit["is_heading"]
        ]

        current_group = []

        # ----------------------------------------------------
        # Start with heading.
        # ----------------------------------------------------

        current_group.extend(
            heading_units
        )

        # ----------------------------------------------------
        # If there is only one content unit,
        # no semantic comparison is needed.
        # ----------------------------------------------------

        if len(content_units) == 1:

            current_group.extend(
                content_units
            )

            final_groups.append(
                current_group
            )

            continue

        # ----------------------------------------------------
        # Encode only content units.
        # ----------------------------------------------------

        texts = [
            unit["text"]
            for unit in content_units
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # ----------------------------------------------------
        # Compare neighboring content only.
        # ----------------------------------------------------

        for index, unit in enumerate(
            content_units
        ):

            if index == 0:

                current_group.append(
                    unit
                )

                continue

            previous_unit = content_units[
                index - 1
            ]

            # ------------------------------------------------
            # FIX:
            # A numbering marker must always stay glued to the
            # text that follows it, regardless of similarity.
            # ------------------------------------------------

            if previous_unit.get(
                "is_numbering",
                False
            ):

                current_group.append(
                    unit
                )

                continue

            previous_embedding = (
                embeddings[index - 1]
            )

            current_embedding = (
                embeddings[index]
            )

            similarity = cosine_similarity(
                previous_embedding,
                current_embedding
            )

            if similarity >= threshold:

                current_group.append(
                    unit
                )

            else:

                # Finish current semantic group.
                if current_group:

                    final_groups.append(
                        current_group
                    )

                # IMPORTANT:
                # Do NOT repeat the heading in every
                # semantic group.
                current_group = [
                    unit
                ]

        if current_group:

            final_groups.append(
                current_group
            )

    return final_groups


# ============================================================
# Size-based splitting
# ============================================================

def split_group_by_size(
    group,
    max_words
):
    """
    Split semantic groups only when they exceed
    the configured maximum size.

    Headings are always kept with the first content
    chunk of their group.
    """

    if not group:
        return []

    chunks = []

    current = []
    current_words = 0

    # --------------------------------------------------------
    # Separate heading from content
    # --------------------------------------------------------

    headings = [
        unit
        for unit in group
        if unit["is_heading"]
    ]

    content = [
        unit
        for unit in group
        if not unit["is_heading"]
    ]

    # --------------------------------------------------------
    # If there is no content, discard.
    # --------------------------------------------------------

    if not content:
        return []

    # --------------------------------------------------------
    # Put heading into first chunk.
    # --------------------------------------------------------

    if headings:

        for heading in headings:

            current.append(
                heading
            )

            current_words += word_count(
                heading["text"]
            )

    # --------------------------------------------------------
    # Process content.
    # --------------------------------------------------------

    for unit in content:

        text = unit["text"]

        words = word_count(text)

        # ----------------------------------------------------
        # Very large single sentence/unit
        # ----------------------------------------------------

        if words > max_words:

            if current:

                chunks.append(
                    current
                )

                current = []
                current_words = 0

            words_list = text.split()

            for start in range(
                0,
                len(words_list),
                max_words
            ):

                piece = " ".join(
                    words_list[
                        start:start + max_words
                    ]
                )

                chunks.append([
                    {
                        "text": piece,
                        "element": unit["element"],
                        "is_heading": False,
                        "is_numbering": False,
                    }
                ])

            continue

        # ----------------------------------------------------
        # Add normally
        # ----------------------------------------------------

        if (
            current_words + words
            <= max_words
        ):

            current.append(
                unit
            )

            current_words += words

        else:

            if current:

                chunks.append(
                    current
                )

            current = [unit]
            current_words = words

    if current:

        chunks.append(
            current
        )

    return chunks


# ============================================================
# Overlap
# ============================================================

def add_overlap(
    chunks,
    overlap_words
):
    """
    Add content overlap between neighboring chunks.

    IMPORTANT:

    Headings are NEVER used as overlap.

    This prevents:

        Heading-only chunks
        repeated headings
        unnecessary duplication

    Overlap is only a context-preservation mechanism.
    """

    if len(chunks) <= 1:
        return chunks

    result = []

    for index, chunk in enumerate(chunks):

        if index == 0:

            result.append(
                chunk
            )

            continue

        previous_chunk = chunks[
            index - 1
        ]

        overlap_units = []

        total_words = 0

        # ----------------------------------------------------
        # Only overlap normal content.
        # ----------------------------------------------------

        content_units = [
            unit
            for unit in previous_chunk
            if not unit["is_heading"]
        ]

        for unit in reversed(
            content_units
        ):

            words = word_count(
                unit["text"]
            )

            if (
                total_words + words
                > overlap_words
            ):
                break

            overlap_units.insert(
                0,
                unit
            )

            total_words += words

        # ----------------------------------------------------
        # Current chunk already owns its heading.
        # Do not duplicate heading from previous chunk.
        # ----------------------------------------------------

        result.append(
            overlap_units + chunk
        )

    return result


# ============================================================
# Build final chunk
# ============================================================

def create_chunk(
    units,
    source_type,
    section,
    chunk_index
):

    texts = [
        unit["text"]
        for unit in units
    ]

    text = " ".join(
        texts
    ).strip()

    elements = [
        unit["element"]
        for unit in units
    ]

    metadata = build_source_metadata(
        elements
    )

    pages = metadata["pages"]

    # --------------------------------------------------------
    # Heading information
    # --------------------------------------------------------

    headings = [
        unit["text"]
        for unit in units
        if unit["is_heading"]
    ]

    # --------------------------------------------------------
    # Chunk ID
    # --------------------------------------------------------

    chunk_id = (
        f"{source_type}_"
        f"{chunk_index:04d}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    # --------------------------------------------------------
    # Final object
    # --------------------------------------------------------

    return {

        "chunk_id": chunk_id,

        "text": text,

        "source_type": source_type,

        "section": section,

        "headings": headings,

        "pages": pages,

        "citation": metadata["citation"],

        "source": {

            "document": (
                metadata["documents"][0]
                if metadata["documents"]
                else None
            ),

            "pages": pages,

            "element_ids": (
                metadata["element_ids"]
            ),

            "bboxes": (
                metadata["bboxes"]
            ),
        },

        "metadata": {

            "source_type": source_type,

            "section": section,

            "chunk_index": chunk_index,

            "word_count": word_count(
                text
            ),

            "page_start": (
                pages[0]
                if pages
                else None
            ),

            "page_end": (
                pages[-1]
                if pages
                else None
            ),
        },
    }


# ============================================================
# Process one section
# ============================================================

def process_section(
    section_elements,
    source_type,
    model,
    config
):

    if not section_elements:
        return []

    section = (
        section_elements[0].get(
            "section",
            "Unknown"
        )
    )

    # --------------------------------------------------------
    # Build semantic units
    # --------------------------------------------------------

    units = build_units(
        section_elements
    )

    if not units:
        return []

    # --------------------------------------------------------
    # Semantic grouping
    # --------------------------------------------------------

    semantic_groups = semantic_group_units(
        units,
        model,
        config["semantic_threshold"]
    )

    # --------------------------------------------------------
    # Size-based splitting
    # --------------------------------------------------------

    size_chunks = []

    for group in semantic_groups:

        pieces = split_group_by_size(
            group,
            config["max_words"]
        )

        size_chunks.extend(
            pieces
        )

    # --------------------------------------------------------
    # Overlap
    # --------------------------------------------------------

    final_chunks = add_overlap(
        size_chunks,
        config["overlap_words"]
    )

    return final_chunks


# ============================================================
# Full document processing
# ============================================================

def process_document(
    elements,
    source_type,
    model
):

    config = CONFIG[
        source_type
    ]

    # --------------------------------------------------------
    # Remove TOC elements.
    # --------------------------------------------------------

    elements = [
        element
        for element in elements
        if not element.get(
            "is_toc",
            False
        )
    ]

    # --------------------------------------------------------
    # Group by section while preserving order.
    # --------------------------------------------------------

    sections = []

    current_section = None
    current_elements = []

    for element in elements:

        section = element.get(
            "section"
        )

        if section is None:

            section = (
                "Document information"
            )

        # ----------------------------------------------------
        # New section
        # ----------------------------------------------------

        if (
            current_section is not None
            and section != current_section
        ):

            sections.append(
                (
                    current_section,
                    current_elements
                )
            )

            current_elements = []

        current_section = section

        current_elements.append(
            element
        )

    if current_elements:

        sections.append(
            (
                current_section,
                current_elements
            )
        )

    # --------------------------------------------------------
    # Process sections independently.
    # --------------------------------------------------------

    final_chunks = []

    chunk_index = 1

    for (
        section_name,
        section_elements
    ) in sections:

        chunk_units = process_section(
            section_elements,
            source_type,
            model,
            config
        )

        for units in chunk_units:

            chunk = create_chunk(
                units,
                source_type,
                section_name,
                chunk_index
            )

            final_chunks.append(
                chunk
            )

            chunk_index += 1

    return final_chunks


# ============================================================
# File selection
# ============================================================

def select_file():

    print(
        "\nAvailable sectioned JSON files:\n"
    )

    files = []

    for source_type in [
        "core",
        "patient"
    ]:

        folder = (
            PROCESSED_DIR /
            source_type
        )

        if not folder.exists():
            continue

        files.extend(
            sorted(
                folder.glob(
                    "*_sections.json"
                )
            )
        )

    if not files:

        raise FileNotFoundError(
            "No *_sections.json files found."
        )

    for index, file in enumerate(
        files,
        start=1
    ):

        print(
            f"{index} - "
            f"{file.parent.name}/"
            f"{file.name}"
        )

    while True:

        choice = input(
            "\nSelect file number: "
        ).strip()

        if choice.isdigit():

            index = int(choice) - 1

            if (
                0 <= index < len(files)
            ):

                return files[index]

        print(
            "Invalid choice."
        )


# ============================================================
# Save
# ============================================================

def save_chunks(
    input_path,
    chunks
):

    output_path = input_path.with_name(
        input_path.stem.replace(
            "_sections",
            ""
        ) + "_chunks.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    return output_path


# ============================================================
# Validation
# ============================================================

def validate_chunks(chunks):

    print(
        "\n" + "=" * 70
    )

    print(
        "CHUNK VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal chunks: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # Source types
    # --------------------------------------------------------

    source_types = sorted(
        set(
            chunk.get(
                "source_type"
            )
            for chunk in chunks
        )
    )

    print(
        f"Source type(s): "
        f"{source_types}"
    )

    # --------------------------------------------------------
    # Missing citation
    # --------------------------------------------------------

    missing_citation = [
        chunk
        for chunk in chunks
        if not chunk.get(
            "citation"
        )
    ]

    if missing_citation:

        print(
            f"\nWARNING: "
            f"{len(missing_citation)} "
            f"chunks have no citation."
        )

    else:

        print(
            "\nOK: Every chunk has citation."
        )

    # --------------------------------------------------------
    # Missing section
    # --------------------------------------------------------

    missing_section = [
        chunk
        for chunk in chunks
        if not chunk.get(
            "section"
        )
    ]

    if missing_section:

        print(
            f"WARNING: "
            f"{len(missing_section)} "
            f"chunks have no section."
        )

    else:

        print(
            "OK: Every chunk has section."
        )

    # --------------------------------------------------------
    # Heading-only chunks
    # --------------------------------------------------------

    heading_only = []

    for chunk in chunks:

        text = normalize_text(
            chunk.get("text", "")
        )

        headings = chunk.get(
            "headings",
            []
        )

        if (
            headings
            and text in headings
        ):

            heading_only.append(
                chunk
            )

    if heading_only:

        print(
            f"WARNING: "
            f"{len(heading_only)} "
            f"heading-only chunks found."
        )

    else:

        print(
            "OK: No heading-only chunks."
        )

    # --------------------------------------------------------
    # Numbering-only chunks
    # --------------------------------------------------------

    numbering_only = [
        chunk
        for chunk in chunks
        if is_numbering_only(
            chunk.get("text", "")
        )
    ]

    if numbering_only:

        print(
            f"WARNING: "
            f"{len(numbering_only)} "
            f"numbering-only chunks found."
        )

    else:

        print(
            "OK: No numbering-only chunks."
        )

    # --------------------------------------------------------
    # Very small chunks
    # --------------------------------------------------------

    tiny_chunks = [
        chunk
        for chunk in chunks
        if chunk.get(
            "metadata",
            {}
        ).get(
            "word_count",
            0
        ) <= 3
    ]

    if tiny_chunks:

        print(
            f"INFO: "
            f"{len(tiny_chunks)} "
            f"very small chunks found."
        )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print(
        "\nFirst 5 chunks:\n"
    )

    for chunk in chunks[:5]:

        print(
            "-" * 70
        )

        print(
            "ID       :",
            chunk["chunk_id"]
        )

        print(
            "Section  :",
            chunk["section"]
        )

        print(
            "Headings :",
            chunk["headings"]
        )

        print(
            "Pages    :",
            chunk["pages"]
        )

        print(
            "Words    :",
            chunk["metadata"][
                "word_count"
            ]
        )

        print(
            "Citation :",
            chunk["citation"]
        )

        print(
            "Text     :",
            chunk["text"][:500]
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "PULMO GUIDE"
    )

    print(
        "HYBRID MEDICAL CHUNKING"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Select sectioned JSON
    # --------------------------------------------------------

    input_path = select_file()

    print(
        f"\nSelected:\n"
        f"{input_path}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        elements = json.load(
            file
        )

    if not elements:

        raise ValueError(
            "Input JSON is empty."
        )

    # --------------------------------------------------------
    # Detect source type
    # --------------------------------------------------------

    source_types = [
        element.get(
            "source_type"
        )
        for element in elements
        if element.get(
            "source_type"
        )
    ]

    source_types = set(
        source_types
    )

    if len(source_types) != 1:

        raise ValueError(
            "Expected exactly one "
            "source_type in the file. "
            f"Found: {source_types}"
        )

    source_type = list(
        source_types
    )[0]

    if source_type not in CONFIG:

        raise ValueError(
            f"Unsupported source_type: "
            f"{source_type}"
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = CONFIG[
        source_type
    ]

    print(
        "\nConfiguration:"
    )

    print(
        f"Source type        : "
        f"{source_type}"
    )

    print(
        f"Max words          : "
        f"{config['max_words']}"
    )

    print(
        f"Overlap words      : "
        f"{config['overlap_words']}"
    )

    print(
        f"Semantic threshold : "
        f"{config['semantic_threshold']}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading semantic model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    print(
        "Creating hybrid medical chunks..."
    )

    chunks = process_document(
        elements,
        source_type,
        model
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = save_chunks(
        input_path,
        chunks
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_chunks(
        chunks
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"SAVED TO:\n"
        f"{output_path}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()

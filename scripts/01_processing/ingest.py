from pathlib import Path
import json

import fitz  # PyMuPDF


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# Core Parser — PyMuPDF
# ============================================================

def parse_core(pdf_path):
    """
    Parse Core guideline using PyMuPDF.

    Steps:
    1. Extract PDF lines.
    2. Merge consecutive lines that belong to the same visual text block.
    3. Preserve formatting metadata for section detection.
    """

    elements = []

    with fitz.open(pdf_path) as doc:

        for page_number, page in enumerate(doc, start=1):

            blocks = page.get_text("dict")["blocks"]

            for block in blocks:

                if block.get("type") != 0:
                    continue

                lines = block.get("lines", [])

                if not lines:
                    continue

                # ------------------------------------------------
                # Convert PDF lines into temporary elements
                # ------------------------------------------------

                block_elements = []

                for line in lines:

                    spans = line.get("spans", [])

                    if not spans:
                        continue

                    text_parts = []

                    for span in spans:

                        span_text = span.get("text", "")

                        if span_text:
                            text_parts.append(span_text)

                    text = "".join(text_parts).strip()

                    if not text:
                        continue

                    # Use first span as formatting reference
                    main_span = spans[0]

                    flags = main_span.get("flags", 0)

                    block_elements.append({
                        "text": text,
                        "type": "Text",
                        "page": page_number,
                        "document": pdf_path.name,
                        "source_type": "core",

                        "font": main_span.get("font"),
                        "size": main_span.get("size"),
                        "flags": flags,
                        "bold": bool(flags & 2),

                        "bbox": line.get("bbox")
                    })

                # ------------------------------------------------
                # Merge lines inside the same PDF block
                # ------------------------------------------------

                i = 0

                while i < len(block_elements):

                    current = block_elements[i]

                    # Start with current line
                    merged_text = current["text"]

                    merged_bbox = list(current["bbox"])

                    j = i + 1

                    while j < len(block_elements):

                        next_element = block_elements[j]

                        # ----------------------------------------
                        # Formatting similarity
                        # ----------------------------------------

                        same_font = (
                            current["font"]
                            == next_element["font"]
                        )

                        same_size = (
                            abs(
                                current["size"]
                                - next_element["size"]
                            ) < 0.5
                        )

                        same_flags = (
                            current["flags"]
                            == next_element["flags"]
                        )

                        # ----------------------------------------
                        # Vertical distance
                        # ----------------------------------------

                        current_bbox = current["bbox"]
                        next_bbox = next_element["bbox"]

                        vertical_gap = (
                            next_bbox[1]
                            - current_bbox[3]
                        )

                        # ----------------------------------------
                        # Determine whether next line
                        # belongs to same visual element
                        # ----------------------------------------

                        should_merge = (
                            same_font
                            and same_size
                            and same_flags
                            and vertical_gap <= current["size"] * 0.8
                        )

                        if not should_merge:
                            break

                        # ----------------------------------------
                        # Merge text
                        # ----------------------------------------

                        merged_text += " " + next_element["text"]

                        # Expand bounding box
                        merged_bbox[0] = min(
                            merged_bbox[0],
                            next_bbox[0]
                        )

                        merged_bbox[1] = min(
                            merged_bbox[1],
                            next_bbox[1]
                        )

                        merged_bbox[2] = max(
                            merged_bbox[2],
                            next_bbox[2]
                        )

                        merged_bbox[3] = max(
                            merged_bbox[3],
                            next_bbox[3]
                        )

                        # Update current reference
                        current = next_element

                        j += 1

                    # --------------------------------------------
                    # Save merged element
                    # --------------------------------------------

                    merged_element = block_elements[i].copy()

                    merged_element["text"] = merged_text.strip()
                    merged_element["bbox"] = merged_bbox

                    elements.append(merged_element)

                    i = j

    return elements

# ============================================================
# Patient Parser — Docling
# ============================================================

def parse_patient(pdf_path):
    """
    Structure-aware parser for patient reports.
    Handles normal text and tables.
    """

    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import TextItem, TableItem

    converter = DocumentConverter()

    result = converter.convert(str(pdf_path))
    document = result.document

    elements = []

    for item, _ in document.iterate_items():

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        if isinstance(item, TextItem):

            text = item.text.strip()

            if not text:
                continue

            page = None

            if item.prov:
                page = item.prov[0].page_no

            elements.append({
                "text": text,
                "type": "Text",
                "page": page,
                "document": pdf_path.name,
                "source_type": "patient"
            })

        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        elif isinstance(item, TableItem):

            try:
                dataframe = item.export_to_dataframe(
                    doc=document
                )
            except Exception:
                continue

            if dataframe.empty:
                continue

            table_text = dataframe.to_string(
                index=False
            ).strip()

            if not table_text:
                continue

            page = None

            if item.prov:
                page = item.prov[0].page_no

            elements.append({
                "text": table_text,
                "type": "Table",
                "page": page,
                "document": pdf_path.name,
                "source_type": "patient"
            })

    return elements


# ============================================================
# Find PDFs
# ============================================================

def find_pdfs(folder):

    return sorted(
        folder.glob("*.pdf")
    )


# ============================================================
# Process PDF
# ============================================================

def process_pdf(pdf_path, source_type):

    print("\n" + "=" * 60)
    print("PULMO GUIDE - INGESTION")
    print("=" * 60)

    print("Document :", pdf_path.name)
    print("Type     :", source_type)

    # --------------------------------------------------------
    # Select parser
    # --------------------------------------------------------

    if source_type == "core":

        print("Parser   : PyMuPDF")

        elements = parse_core(pdf_path)

    else:

        print("Parser   : Docling")

        elements = parse_patient(pdf_path)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = PROCESSED_DIR / source_type

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / (
        pdf_path.stem + ".json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            elements,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    text_count = sum(
        x["type"] == "Text"
        for x in elements
    )

    table_count = sum(
        x["type"] == "Table"
        for x in elements
    )

    pages = sorted({
        x["page"]
        for x in elements
        if x["page"] is not None
    })

    print("\nCompleted successfully.")
    print("Elements :", len(elements))
    print("Text     :", text_count)
    print("Tables   :", table_count)
    print("Pages    :", len(pages))
    print("Saved to :", output_path)

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print("\nPreview:")

    for element in elements[:5]:

        print("\n------------------------------")
        print("Type :", element["type"])
        print("Page :", element["page"])
        print("Text :", element["text"][:300])


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("PULMO GUIDE")
    print("PDF INGESTION")
    print("=" * 60)

    print("\nSelect document type:")
    print("1 - Core Knowledge Base")
    print("2 - Patient Report")

    choice = input("\nEnter choice (1/2): ").strip()

    # --------------------------------------------------------
    # Determine source
    # --------------------------------------------------------

    if choice == "1":

        source_type = "core"
        folder = RAW_DIR / "core"

    elif choice == "2":

        source_type = "patient"
        folder = RAW_DIR / "patient"

    else:

        print("\nInvalid choice.")
        return

    # --------------------------------------------------------
    # Find PDFs
    # --------------------------------------------------------

    pdfs = find_pdfs(folder)

    if not pdfs:

        print(
            f"\nNo PDF files found in:\n{folder}"
        )

        return

    # --------------------------------------------------------
    # If multiple PDFs exist
    # --------------------------------------------------------

    if len(pdfs) > 1:

        print("\nAvailable PDFs:\n")

        for i, pdf in enumerate(pdfs, start=1):

            print(f"{i}. {pdf.name}")

        selected = input(
            "\nSelect PDF number: "
        ).strip()

        try:
            index = int(selected) - 1
            pdf_path = pdfs[index]

        except (ValueError, IndexError):

            print("\nInvalid selection.")
            return

    else:

        pdf_path = pdfs[0]

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    process_pdf(
        pdf_path,
        source_type
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
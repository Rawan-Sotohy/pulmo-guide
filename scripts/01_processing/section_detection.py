from pathlib import Path
import json
import re


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# NICE known structural headings
# ============================================================

NICE_MAJOR_HEADINGS = {
    "Overview",
    "Diagnosis and staging",
    "Management",
    "Palliative interventions and supportive and palliative care",
    "Follow-up and patient perspectives",
    "Recommendations for research",
    "Finding more information and committee details",
    "Update information",
}


# ============================================================
# NICE heading patterns
# ============================================================

NUMBERED_HEADING = re.compile(
    r"^\d+(?:\.\d+)*\s+.+$"
)


# ============================================================
# Text helpers
# ============================================================

def normalize(text):

    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_toc_line(text):
    """
    Removes TOC dotted leaders and page numbers.

    Example:

    Overview....................................5

    becomes:

    Overview
    """

    text = normalize(text)

    # Remove dotted leaders + page number
    text = re.sub(
        r"\.{3,}\s*\d+\s*$",
        "",
        text
    ).strip()

    return text


# ============================================================
# TOC extraction
# ============================================================

def extract_toc_sections(elements):
    """
    Extract section names from NICE Table of Contents.

    NICE TOC is expected on pages 3-4.

    The TOC is NOT treated as medical content.
    It is only used as a reference list of valid section names.
    """

    toc_sections = set()

    for element in elements:

        page = element.get("page", 0)

        # NICE TOC pages
        if page not in (3, 4):
            continue

        text = normalize(
            element.get("text", "")
        )

        if not text:
            continue

        original_text = text

        cleaned = clean_toc_line(text)

        if not cleaned:
            continue

        # Ignore Contents title
        if cleaned.lower() == "contents":
            continue

        # ----------------------------------------------------
        # Ignore obvious non-section TOC entries
        # ----------------------------------------------------

        ignored_entries = {
            "Who is it for?",
            "Information for the public",
            "Information for healthcare professionals",
        }

        if cleaned in ignored_entries:
            continue

        # ----------------------------------------------------
        # Known NICE major headings
        # ----------------------------------------------------

        if cleaned in NICE_MAJOR_HEADINGS:

            toc_sections.add(cleaned)

            continue

        # ----------------------------------------------------
        # Numbered sections
        #
        # Example:
        # 1.2 Diagnosis
        # 1.2.1 Something
        # ----------------------------------------------------

        if NUMBERED_HEADING.match(cleaned):

            toc_sections.add(cleaned)

            continue

        # ----------------------------------------------------
        # Unnumbered subsection headings
        #
        # Only accept these if the original TOC line
        # actually had dotted page-number decoration.
        # ----------------------------------------------------

        if re.search(
            r"\.{3,}\s*\d+\s*$",
            original_text
        ):

            toc_sections.add(cleaned)

    return toc_sections


# ============================================================
# NICE heading detection
# ============================================================

def is_nice_heading(element):
    """
    Detect NICE headings using:

        - text structure
        - font
        - font size
        - parser metadata
        - page layout

    Ordinary paragraphs are NOT classified as headings
    simply because they are short.
    """

    text = normalize(
        element.get("text", "")
    )

    if not text:
        return False

    page = element.get("page", 0)

    font = str(
        element.get("font", "")
    ).lower()

    size = float(
        element.get("size", 0) or 0
    )

    flags = int(
        element.get("flags", 0) or 0
    )

    bold = bool(
        element.get("bold", False)
    )

    # --------------------------------------------------------
    # Ignore front matter
    # --------------------------------------------------------

    if page <= 4:
        return False

    # --------------------------------------------------------
    # Basic cleanup
    # --------------------------------------------------------

    cleaned = clean_toc_line(text)

    if not cleaned:
        return False

    # --------------------------------------------------------
    # Existing parser heading flag
    # --------------------------------------------------------

    if element.get("is_section_heading") is True:

        return True

    # --------------------------------------------------------
    # Known NICE major headings
    # --------------------------------------------------------

    if cleaned in NICE_MAJOR_HEADINGS:

        return True

    # --------------------------------------------------------
    # Numbered NICE headings
    # --------------------------------------------------------

    if NUMBERED_HEADING.match(cleaned):

        heading_font = (
            "lora" in font
            or "semibold" in font
            or "bold" in font
        )

        if size >= 13:

            return True

        if heading_font and size >= 12:

            return True

    # --------------------------------------------------------
    # Unnumbered subsection headings
    # --------------------------------------------------------

    heading_font = (
        "lora" in font
        or "semibold" in font
        or "bold" in font
    )

    if heading_font and size >= 13:

        if len(cleaned) <= 180:

            if not cleaned.endswith(
                (".", ":", ";", "?")
            ):

                return True

    # --------------------------------------------------------
    # Large text
    # --------------------------------------------------------

    if size >= 16 and len(cleaned) <= 180:

        if not cleaned.endswith(
            (".", ":", ";", "?")
        ):

            return True

    return False


# ============================================================
# Heading classification
# ============================================================

def classify_heading(text):
    """
    Gives structural information only.

    No level hierarchy is used for section assignment.
    """

    text = normalize(text)

    if NUMBERED_HEADING.match(text):

        return "numbered"

    if text in NICE_MAJOR_HEADINGS:

        return "major"

    return "subsection"


# ============================================================
# Patient section detection
# ============================================================

PATIENT_HEADINGS = {
    "PATHOLOGY",
    "MOLECULAR",
    "IMAGING",
    "RADIOLOGY",
    "FINDINGS",
    "IMPRESSION",
    "CONCLUSION",
    "INTERPRETATION",
    "DIAGNOSIS",
    "CLINICAL HISTORY",
    "HISTORY",
    "INDICATION",
    "INDICATIONS",
    "PROCEDURE",
    "TECHNIQUE",
    "SPECIMEN",
    "MICROSCOPIC DESCRIPTION",
    "GROSS DESCRIPTION",
    "FINAL DIAGNOSIS",
    "SPIROMETRY",
    "DIFFUSION CAPACITY",
    "RESULTS",
    "RECOMMENDATION",
    "RECOMMENDATIONS",
}


def is_patient_heading(element):
    """
    Detect patient-report headings.
    """

    text = normalize(
        element.get("text", "")
    )

    if not text:
        return False

    upper = text.upper()

    # --------------------------------------------------------
    # Exact heading
    # --------------------------------------------------------

    if upper in PATIENT_HEADINGS:

        return True

    # --------------------------------------------------------
    # Heading followed by colon
    # --------------------------------------------------------

    for heading in PATIENT_HEADINGS:

        if upper.startswith(
            heading + ":"
        ):

            remainder = upper[
                len(heading) + 1:
            ].strip()

            if not remainder:

                return True

    # --------------------------------------------------------
    # Typography fallback
    # --------------------------------------------------------

    font = str(
        element.get("font", "")
    ).lower()

    size = float(
        element.get("size", 0) or 0
    )

    heading_font = (
        "bold" in font
        or "semibold" in font
        or "heavy" in font
    )

    if heading_font and size >= 12:

        if len(text) <= 100 and not text.endswith("."):

            return True

    return False


# ============================================================
# Select input file
# ============================================================

def select_file(source_type):

    folder = PROCESSED_DIR / source_type

    if not folder.exists():

        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )

    files = sorted(
        [
            file
            for file in folder.glob("*_cleaned.json")
            if "_sections" not in file.stem
        ]
    )

    if not files:

        raise FileNotFoundError(
            f"No cleaned JSON files found in:\n{folder}"
        )

    print("\nAvailable cleaned files:\n")

    for i, file in enumerate(files, 1):

        print(
            f"{i} - {file.name}"
        )

    while True:

        choice = input(
            "\nSelect file number: "
        ).strip()

        if choice.isdigit():

            index = int(choice) - 1

            if 0 <= index < len(files):

                return files[index]

        print("Invalid choice.")


# ============================================================
# Process NICE
# ============================================================

def process_nice(elements):

    result = []

    current_section = None

    # ========================================================
    # Extract TOC reference
    # ========================================================

    toc_sections = extract_toc_sections(
        elements
    )

    print(
        f"\nTOC sections found: "
        f"{len(toc_sections)}"
    )

    # --------------------------------------------------------
    # Optional preview of TOC
    # --------------------------------------------------------

    print("\nTOC reference sections:")

    for section in sorted(
        toc_sections
    )[:30]:

        print(
            f" - {section}"
        )

    if len(toc_sections) > 30:

        print(
            f" ... and "
            f"{len(toc_sections) - 30} more"
        )

    # ========================================================
    # Process elements
    # ========================================================

    for element in elements:

        new_element = element.copy()

        text = normalize(
            element.get("text", "")
        )

        page = element.get(
            "page",
            0
        )

        # ====================================================
        # FRONT MATTER
        # ====================================================

        if page == 1:

            current_section = (
                "Document information"
            )

            new_element[
                "is_section_heading"
            ] = False

            new_element[
                "section"
            ] = current_section

            new_element[
                "section_type"
            ] = "front_matter"

            result.append(new_element)

            continue

        if page == 2:

            current_section = (
                "Guideline responsibility"
            )

            new_element[
                "is_section_heading"
            ] = False

            new_element[
                "section"
            ] = current_section

            new_element[
                "section_type"
            ] = "front_matter"

            result.append(new_element)

            continue

        if page in (3, 4):

    # TOC is used only as a reference for section detection.
    # It must NOT become retrievable medical content.

            new_element[
            "is_section_heading"
            ] = False

            new_element[
            "section"
            ] = "Table of contents"

            new_element[
           "section_type"
           ] = "toc"

            new_element[
           "is_toc"
            ] = True

            result.append(new_element)

            continue

        # ====================================================
        # MEDICAL CONTENT
        # Page 5+
        # ====================================================

        cleaned = clean_toc_line(
            text
        )

        detected = False

        # ----------------------------------------------------
        # Existing reliable heading detector
        # ----------------------------------------------------

        if is_nice_heading(element):

            detected = True

        # ----------------------------------------------------
        # TOC confirmation
        #
        # If the exact cleaned text exists in the TOC,
        # it is a valid section heading.
        # ----------------------------------------------------

        if cleaned in toc_sections:

            detected = True

        # ----------------------------------------------------
        # NEW SECTION
        # ----------------------------------------------------

        if detected and cleaned:

            current_section = cleaned

            new_element[
                "is_section_heading"
            ] = True

            new_element[
                "section"
            ] = current_section

            new_element[
                "section_type"
            ] = classify_heading(
                current_section
            )

        # ----------------------------------------------------
        # CONTENT
        # ----------------------------------------------------

        else:

            # Every medical element must have a section.
            # Normally current_section will already exist.
            # This is only a safety fallback.

            if current_section is None:

                current_section = (
                    "Medical content"
                )

            new_element[
                "is_section_heading"
            ] = False

            new_element[
                "section"
            ] = current_section

            new_element[
                "section_type"
            ] = "content"

        result.append(new_element)

    return result


# ============================================================
# Process Patient
# ============================================================

def process_patient(elements):

    result = []

    current_section = (
        "Document information"
    )

    for element in elements:

        new_element = element.copy()

        text = normalize(
            element.get("text", "")
        )

        if is_patient_heading(element):

            upper = text.upper()

            detected_name = None

            for heading in PATIENT_HEADINGS:

                if (
                    upper == heading
                    or upper.startswith(
                        heading + ":"
                    )
                ):

                    detected_name = heading

                    break

            if detected_name is None:

                detected_name = text

            current_section = (
                detected_name
            )

            new_element[
                "is_section_heading"
            ] = True

            new_element[
                "section"
            ] = current_section

            new_element[
                "section_type"
            ] = "patient_section"

        else:

            new_element[
                "is_section_heading"
            ] = False

            new_element[
                "section"
            ] = current_section

            new_element[
                "section_type"
            ] = "content"

        result.append(new_element)

    return result


# ============================================================
# Save
# ============================================================

def save_result(
    input_path,
    result
):

    output_path = input_path.with_name(
        input_path.stem
        + "_sections.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )

    return output_path


# ============================================================
# Validation
# ============================================================

def print_validation(result):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SECTION DETECTION VALIDATION"
    )

    print(
        "=" * 70
    )

    sections = []

    for element in result:

        section = element.get(
            "section"
        )

        if (
            section
            and section not in sections
        ):

            sections.append(section)

    print(
        f"\nTotal elements : "
        f"{len(result)}"
    )

    print(
        f"Unique sections: "
        f"{len(sections)}"
    )

    print(
        "\nDetected sections:\n"
    )

    for i, section in enumerate(
        sections,
        1
    ):

        print(
            f"{i:03d}. {section}"
        )

    # ========================================================
    # Check NULL sections
    # ========================================================

    null_sections = [
        element
        for element in result
        if not element.get("section")
    ]

    print(
        "\n"
        + "-" * 70
    )

    if null_sections:

        print(
            f"ERROR: "
            f"{len(null_sections)} elements "
            f"have empty section!"
        )

    else:

        print(
            "OK: Every element has a section."
        )

    # ========================================================
    # Print headings
    # ========================================================

    headings = [
        element
        for element in result
        if element.get(
            "is_section_heading"
        )
    ]

    print(
        f"\nDetected headings: "
        f"{len(headings)}"
    )

    print(
        "\nHeadings:\n"
    )

    for heading in headings:

        print(
            f"Page "
            f"{heading.get('page'):>2} | "
            f"{heading.get('text')}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PULMO GUIDE"
    )

    print(
        "SECTION DETECTION + TOC"
    )

    print(
        "=" * 70
    )

    print(
        "\nSelect document type:"
    )

    print(
        "1 - NICE Core Knowledge"
    )

    print(
        "2 - Patient Report"
    )

    while True:

        choice = input(
            "\nEnter choice (1/2): "
        ).strip()

        if choice == "1":

            source_type = "core"

            break

        elif choice == "2":

            source_type = "patient"

            break

        print(
            "Invalid choice."
        )

    # ========================================================
    # Select cleaned JSON
    # ========================================================

    input_path = select_file(
        source_type
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROCESSING"
    )

    print(
        "=" * 70
    )

    print(
        f"\nInput : "
        f"{input_path.name}"
    )

    print(
        f"Type  : "
        f"{source_type}"
    )

    # ========================================================
    # Load
    # ========================================================

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        elements = json.load(
            file
        )

    # ========================================================
    # Detect
    # ========================================================

    if source_type == "core":

        result = process_nice(
            elements
        )

    else:

        result = process_patient(
            elements
        )

    # ========================================================
    # Save
    # ========================================================

    output_path = save_result(
        input_path,
        result
    )

    # ========================================================
    # Validation
    # ========================================================

    print_validation(
        result
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"SAVED TO:\n"
        f"{output_path}"
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
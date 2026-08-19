from pathlib import Path
import json
import re


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# Core Constants
# ============================================================

CORE_HEADER = "Lung cancer: diagnosis and management (NG122)"


# ============================================================
# Basic Text Normalization
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize PDF formatting without changing medical content.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize tabs and repeated spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# Core Cleaning
# ============================================================
def is_core_header(text: str) -> bool:
    """
    Detect repeated NICE document header/footer title.
    """

    return text.strip() == CORE_HEADER


def is_nice_footer(text: str) -> bool:
    """
    Detect NICE copyright/footer fragments.
    """

    stripped = text.strip()

    return (
        stripped.startswith("© NICE")
        or "Subject to Notice of rights" in stripped
        or "notice-of-rights" in stripped.lower()
    )


def is_page_marker(text: str) -> bool:
    """
    Detect page number fragments such as:

        Page 2 of
        50
    """

    return bool(
        re.fullmatch(
            r"Page\s+\d+\s+of",
            text.strip(),
            re.IGNORECASE
        )
    )



def clean_core(elements):
    """
    Clean NICE Core Knowledge Base.

    Removes:
    - repeated NICE header
    - NICE copyright footer
    - page footer markers

    Preserves:
    - medical content
    - numbers
    - recommendations
    - sections
    - page metadata
    """

    cleaned = []

    skip_next_total_page_number = False

    for index, element in enumerate(elements):

        text = element.get("text", "")

        if not text.strip():
            continue

        stripped = text.strip()

        # ----------------------------------------------------
        # Repeated header
        # ----------------------------------------------------

        if is_core_header(stripped):
            continue

        # ----------------------------------------------------
        # NICE copyright footer
        # ----------------------------------------------------

        if is_nice_footer(stripped):
            continue

        # ----------------------------------------------------
        # Page X of
        # ----------------------------------------------------

        if is_page_marker(stripped):

            # The next element may be the total page count,
            # e.g. "50".
            #
            # We remove it ONLY when it immediately follows
            # "Page X of" on the same PDF page.

            skip_next_total_page_number = True

            continue

        # ----------------------------------------------------
        # Total page count
        # ----------------------------------------------------

        if skip_next_total_page_number:

            previous_element = (
                elements[index - 1]
                if index > 0
                else None
            )

            previous_text = (
                previous_element.get("text", "").strip()
                if previous_element
                else ""
            )

            same_page = (
                previous_element is not None
                and previous_element.get("page")
                == element.get("page")
            )

            is_total_page_number = bool(
                re.fullmatch(r"\d+", stripped)
            )

            if (
                same_page
                and is_total_page_number
                and is_page_marker(previous_text)
            ):
                skip_next_total_page_number = False
                continue

            # Safety reset:
            # if the next element wasn't actually the total
            # page count, keep it.
            skip_next_total_page_number = False

        # ----------------------------------------------------
        # Normalize actual content
        # ----------------------------------------------------

        normalized = normalize_text(text)

        if not normalized:
            continue

        cleaned_element = element.copy()
        cleaned_element["text"] = normalized

        cleaned.append(cleaned_element)

    return cleaned


# ============================================================
# Patient Cleaning
# ============================================================

def is_patient_decorative_line(text: str) -> bool:
    """
    Remove only obvious decorative separators.

    Examples:
        ~~~~~~~~~~~~~
        ---------------
        =================

    No medical content is removed.
    """

    stripped = text.strip()

    if not stripped:
        return True

    return bool(
        re.fullmatch(
            r"[~=_\-]{5,}",
            stripped
        )
    )


def clean_patient(elements):
    """
    Conservative cleaning for Patient Reports.

    Preserves:
    - medical values
    - numbers
    - percentages
    - dates
    - units
    - sections
    - tables
    - page metadata

    Only obvious decorative lines are removed.
    """

    cleaned = []

    for element in elements:

        text = element.get("text", "")

        if not text.strip():
            continue

        # ----------------------------------------------------
        # Tables
        # ----------------------------------------------------

        if element.get("type") == "Table":

            cleaned_element = element.copy()

            cleaned_element["text"] = normalize_text(text)

            if cleaned_element["text"]:
                cleaned.append(cleaned_element)

            continue

        # ----------------------------------------------------
        # Decorative lines
        # ----------------------------------------------------

        if is_patient_decorative_line(text):
            continue

        # ----------------------------------------------------
        # Normal patient content
        # ----------------------------------------------------

        normalized = normalize_text(text)

        if not normalized:
            continue

        cleaned_element = element.copy()
        cleaned_element["text"] = normalized

        cleaned.append(cleaned_element)

    return cleaned


# ============================================================
# Process File
# ============================================================

def process_file(input_path: Path, source_type: str):

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        elements = json.load(file)

    if source_type == "core":

        cleaned = clean_core(elements)

    elif source_type == "patient":

        cleaned = clean_patient(elements)

    else:

        raise ValueError(
            f"Unsupported source type: {source_type}"
        )

    # Save cleaned version beside original
    output_path = input_path.with_name(
        input_path.stem + "_cleaned.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cleaned,
            file,
            ensure_ascii=False,
            indent=2
        )

    return output_path, elements, cleaned


# ============================================================
# Select File
# ============================================================

def select_file(source_type: str):

    folder = PROCESSED_DIR / source_type

    files = sorted(
        [
            file
            for file in folder.glob("*.json")
            if not file.stem.endswith("_cleaned")
        ]
    )

    if not files:

        raise FileNotFoundError(
            f"No processed JSON files found in: {folder}"
        )

    print("\nAvailable files:")

    for index, file in enumerate(files, start=1):

        print(
            f"{index} - {file.name}"
        )

    while True:

        choice = input(
            "\nSelect file number: "
        ).strip()

        if choice.isdigit():

            index = int(choice) - 1

            if 0 <= index < len(files):

                return files[index]

        print("Invalid choice. Try again.")


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("PULMO GUIDE")
    print("TEXT CLEANING")
    print("=" * 60)

    print("\nSelect document type:")
    print("1 - Core Knowledge Base")
    print("2 - Patient Report")

    while True:

        choice = input(
            "\nEnter choice (1/2): "
        ).strip()

        if choice == "1":

            source_type = "core"
            break

        if choice == "2":

            source_type = "patient"
            break

        print("Invalid choice.")

    input_path = select_file(source_type)

    print("\n" + "=" * 60)
    print("PULMO GUIDE - CLEANING")
    print("=" * 60)

    print(
        f"Document : {input_path.name}"
    )

    print(
        f"Type     : {source_type}"
    )

    output_path, original, cleaned = process_file(
        input_path,
        source_type
    )

    original_tables = sum(
        1
        for element in original
        if element.get("type") == "Table"
    )

    cleaned_tables = sum(
        1
        for element in cleaned
        if element.get("type") == "Table"
    )

    print("\nCompleted successfully.")

    print(
        f"Before   : {len(original)} elements"
    )

    print(
        f"After    : {len(cleaned)} elements"
    )

    print(
        f"Removed  : {len(original) - len(cleaned)} elements"
    )

    print(
        f"Tables   : {original_tables} -> {cleaned_tables}"
    )

    print(
        f"Saved to : {output_path}"
    )

    print("\nPreview:")

    for element in cleaned[:5]:

        print("\n------------------------------")

        print(
            "Type :",
            element.get("type")
        )

        print(
            "Page :",
            element.get("page")
        )

        print(
            "Text :",
            element.get("text", "")[:300]
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
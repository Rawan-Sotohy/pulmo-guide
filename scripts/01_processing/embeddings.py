from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# Embedding Model
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ============================================================
# Select chunks file
# ============================================================

def select_file():

    print("\nAvailable chunk JSON files:\n")

    files = []

    for source_type in [
        "core",
        "patient"
    ]:

        folder = PROCESSED_DIR / source_type

        if not folder.exists():
            continue

        files.extend(
            sorted(
                folder.glob("*_chunks.json")
            )
        )

    if not files:

        raise FileNotFoundError(
            "No *_chunks.json files found."
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

            if 0 <= index < len(files):

                return files[index]

        print("Invalid choice.")


# ============================================================
# Generate embeddings
# ============================================================

def generate_embeddings(
    chunks,
    model
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"\nGenerating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


# ============================================================
# Build output
# ============================================================

def build_embedding_data(
    chunks,
    embeddings
):

    output = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        output.append({

            "chunk_id": chunk["chunk_id"],

            "text": chunk["text"],

            "embedding": embedding.tolist(),

            "source_type": chunk["source_type"],

            # ------------------------------------------------
            # Preserve patient session identity
            # ------------------------------------------------

            "session_id": chunk.get(
                "session_id"
            ),

            "section": chunk["section"],

            "headings": chunk["headings"],

            "pages": chunk["pages"],

            "citation": chunk["citation"],

            "source": chunk["source"],

            # ------------------------------------------------
            # Existing chunk metadata
            # This already contains session_id as well.
            # ------------------------------------------------

            "metadata": chunk["metadata"],
        })

    return output


# ============================================================
# Save
# ============================================================

def save_embeddings(
    input_path,
    data
):

    output_path = input_path.with_name(
        input_path.stem.replace(
            "_chunks",
            ""
        ) + "_embeddings.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )

    return output_path


# ============================================================
# Validation
# ============================================================

def validate_embeddings(
    data
):

    print(
        "\n" + "=" * 70
    )

    print("EMBEDDING VALIDATION")

    print("=" * 70)

    print(
        f"\nTotal chunks: "
        f"{len(data)}"
    )

    if not data:

        raise ValueError(
            "No embedding data generated."
        )

    # ========================================================
    # Source type validation
    # ========================================================

    source_types = set(
        item.get("source_type")
        for item in data
    )

    print(
        f"Source type(s): "
        f"{source_types}"
    )

    if len(source_types) != 1:

        raise ValueError(
            "Expected exactly one source_type "
            f"in the embedding file. "
            f"Found: {source_types}"
        )

    source_type = next(
        iter(source_types)
    )

    # ========================================================
    # Embedding dimension validation
    # ========================================================

    dimensions = set(
        len(item["embedding"])
        for item in data
    )

    print(
        f"Embedding dimensions: "
        f"{dimensions}"
    )

    if dimensions != {384}:

        raise ValueError(
            "Unexpected embedding dimension. "
            f"Found: {dimensions}"
        )

    # ========================================================
    # Missing embeddings
    # ========================================================

    missing_embeddings = [
        item
        for item in data
        if not item.get("embedding")
    ]

    if missing_embeddings:

        raise ValueError(
            f"{len(missing_embeddings)} "
            "chunks have missing embeddings."
        )

    # ========================================================
    # Duplicate chunk IDs
    # ========================================================

    chunk_ids = [
        item["chunk_id"]
        for item in data
    ]

    if len(chunk_ids) != len(set(chunk_ids)):

        raise ValueError(
            "Duplicate chunk IDs found."
        )

    # ========================================================
    # Patient session validation
    # ========================================================

    if source_type == "patient":

        session_ids = {
            item.get("session_id")
            for item in data
        }

        print(
            f"Patient session ID(s): "
            f"{session_ids}"
        )

        # Every patient embedding must have a session ID.
        if None in session_ids:

            raise ValueError(
                "Patient embeddings contain chunks "
                "without a session_id."
            )

        # All chunks from one processed file must
        # belong to the same session.
        if len(session_ids) != 1:

            raise ValueError(
                "Multiple session IDs found in "
                "the same patient embedding file. "
                f"Found: {session_ids}"
            )

        # Make sure metadata also preserves
        # the same session ID.
        session_id = next(
            iter(session_ids)
        )

        for item in data:

            metadata_session_id = (
                item.get(
                    "metadata",
                    {}
                ).get(
                    "session_id"
                )
            )

            if metadata_session_id != session_id:

                raise ValueError(
                    "Session ID mismatch between "
                    "chunk and metadata."
                )

        print(
            "OK: Every patient chunk has the "
            "same valid session_id."
        )

        print(
            "OK: Session ID is preserved "
            "inside chunk metadata."
        )

    # ========================================================
    # Core validation
    # ========================================================

    else:

        print(
            "OK: Core knowledge embeddings "
            "do not require a patient session_id."
        )

    # ========================================================
    # General validation success
    # ========================================================

    print(
        "\nOK: Every chunk has a "
        "384-dimensional embedding."
    )

    print(
        "OK: No duplicate chunk IDs."
    )

    # ========================================================
    # Preview
    # ========================================================

    print(
        "\nFirst embedding preview:"
    )

    print(
        data[0]["embedding"][:10]
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print("PULMO GUIDE")

    print("EMBEDDING GENERATION")

    print("=" * 70)

    # --------------------------------------------------------
    # Select chunks file
    # --------------------------------------------------------

    input_path = select_file()

    print(
        f"\nSelected:\n{input_path}"
    )

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:

        raise ValueError(
            "Chunks JSON is empty."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = generate_embeddings(
        chunks,
        model
    )

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    data = build_embedding_data(
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = save_embeddings(
        input_path,
        data
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_embeddings(
        data
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"SAVED TO:\n{output_path}"
    )

    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()
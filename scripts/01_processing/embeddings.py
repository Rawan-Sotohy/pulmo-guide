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

def generate_embeddings(chunks, model):

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

def build_embedding_data(chunks, embeddings):

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

            "section": chunk["section"],

            "headings": chunk["headings"],

            "pages": chunk["pages"],

            "citation": chunk["citation"],

            "source": chunk["source"],

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

def validate_embeddings(data):

    print(
        "\n" + "=" * 70
    )

    print("EMBEDDING VALIDATION")

    print("=" * 70)

    print(
        f"\nTotal chunks: {len(data)}"
    )

    if not data:
        raise ValueError(
            "No embedding data generated."
        )

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

    chunk_ids = [
        item["chunk_id"]
        for item in data
    ]

    if len(chunk_ids) != len(set(chunk_ids)):

        raise ValueError(
            "Duplicate chunk IDs found."
        )

    print(
        "OK: Every chunk has a 384-dimensional embedding."
    )

    print(
        "OK: No duplicate chunk IDs."
    )

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

    print("=" * 70
    )

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


if __name__ == "__main__":
    main()

import json
import chromadb
from pathlib import Path


# ============================================================
# PULMO GUIDE
# CHROMADB VECTOR STORE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

VECTOR_DB_DIR = (
    BASE_DIR
    / "data"
    / "vector_store"
)

COLLECTION_NAME = "pulmo_guide"


# ============================================================
# Find embedding files
# ============================================================

def find_embedding_files():

    print("\nSearching for embedding files...\n")

    files = []

    for source_type in [
        "core",
        "patient"
    ]:

        folder = (
            PROCESSED_DIR
            / source_type
        )

        if not folder.exists():
            continue

        files.extend(
            sorted(
                folder.glob(
                    "*_embeddings.json"
                )
            )
        )

    if not files:

        raise FileNotFoundError(
            "No *_embeddings.json files found."
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

    return files


# ============================================================
# Load embeddings
# ============================================================

def load_embeddings(files):

    all_chunks = []

    print("\nLoading embeddings...")

    for file in files:

        print(
            f"\nLoading: "
            f"{file.parent.name}/"
            f"{file.name}"
        )

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

        if not chunks:

            print(
                "WARNING: File is empty."
            )

            continue

        print(
            f"Loaded {len(chunks)} chunks."
        )

        all_chunks.extend(
            chunks
        )

    if not all_chunks:

        raise ValueError(
            "No embedding data loaded."
        )

    return all_chunks


# ============================================================
# Create ChromaDB
# ============================================================

def create_collection():

    print(
        "\nInitializing ChromaDB..."
    )

    client = chromadb.PersistentClient(
        path=str(
            VECTOR_DB_DIR
        )
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description":
                "Pulmo Guide Medical Knowledge Base"
        }
    )

    return client, collection


# ============================================================
# Prepare data
# ============================================================

def prepare_data(chunks):

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:

        ids.append(
            chunk["chunk_id"]
        )

        documents.append(
            chunk["text"]
        )

        embeddings.append(
            chunk["embedding"]
        )

        metadata = chunk.get(
            "metadata",
            {}
        )

        # ----------------------------------------------------
        # Chroma metadata values must be simple types
        # ----------------------------------------------------

        metadatas.append({

            "source_type":
                chunk.get(
                    "source_type",
                    metadata.get(
                        "source_type",
                        "unknown"
                    )
                ),

            "section":
                chunk.get(
                    "section",
                    metadata.get(
                        "section",
                        ""
                    )
                ),

            "page_start":
                metadata.get(
                    "page_start",
                    0
                ),

            "page_end":
                metadata.get(
                    "page_end",
                    0
                ),

            "word_count":
                metadata.get(
                    "word_count",
                    0
                ),

            "citation":
                chunk.get(
                    "citation",
                    ""
                )
        })

    return (
        ids,
        documents,
        embeddings,
        metadatas
    )


# ============================================================
# Validate input
# ============================================================

def validate_input(
    ids,
    documents,
    embeddings,
    metadatas
):

    print(
        "\nValidating input data..."
    )

    # --------------------------------------------------------
    # Same lengths
    # --------------------------------------------------------

    lengths = {
        len(ids),
        len(documents),
        len(embeddings),
        len(metadatas)
    }

    if len(lengths) != 1:

        raise ValueError(
            "IDs, documents, embeddings, "
            "and metadata lengths do not match."
        )

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    if len(ids) != len(set(ids)):

        raise ValueError(
            "Duplicate chunk IDs found."
        )

    # --------------------------------------------------------
    # Embedding dimensions
    # --------------------------------------------------------

    dimensions = set(
        len(embedding)
        for embedding in embeddings
    )

    print(
        f"Embedding dimensions: "
        f"{dimensions}"
    )

    if dimensions != {384}:

        raise ValueError(
            "Unexpected embedding dimension. "
            f"Expected {{384}}, found {dimensions}"
        )

    # --------------------------------------------------------
    # Missing data
    # --------------------------------------------------------

    if any(
        not document.strip()
        for document in documents
    ):

        raise ValueError(
            "One or more chunks have empty text."
        )

    print(
        "OK: Input data is valid."
    )


# ============================================================
# Add data to ChromaDB
# ============================================================

def add_to_collection(
    collection,
    ids,
    documents,
    embeddings,
    metadatas
):

    print(
        "\nAdding chunks to ChromaDB..."
    )

    collection.upsert(

        ids=ids,

        documents=documents,

        embeddings=embeddings,

        metadatas=metadatas
    )

    print(
        f"OK: {len(ids)} chunks "
        "added/updated successfully."
    )


# ============================================================
# Validation
# ============================================================

def validate_collection(
    collection,
    expected_count
):

    print(
        "\n" + "=" * 70
    )

    print(
        "VECTOR STORE VALIDATION"
    )

    print(
        "=" * 70
    )

    count = collection.count()

    print(
        f"Chunks stored: "
        f"{count}"
    )

    print(
        f"Expected chunks: "
        f"{expected_count}"
    )

    if count == expected_count:

        print(
            "OK: All chunks stored successfully."
        )

    else:

        print(
            "WARNING: Stored chunk count "
            "does not match input count."
        )

    # --------------------------------------------------------
    # Check source types
    # --------------------------------------------------------

    result = collection.get(
        include=[
            "metadatas"
        ]
    )

    source_types = set()

    for metadata in result.get(
        "metadatas",
        []
    ):

        if metadata:

            source_type = metadata.get(
                "source_type"
            )

            if source_type:

                source_types.add(
                    source_type
                )

    print(
        f"Source types stored: "
        f"{sorted(source_types)}"
    )

    # --------------------------------------------------------
    # Test retrieval
    # --------------------------------------------------------

    print(
        "\nTesting vector database..."
    )

    test_result = collection.get(
        limit=1,
        include=[
            "documents",
            "metadatas"
        ]
    )

    if test_result["ids"]:

        print(
            "OK: ChromaDB retrieval is working."
        )

        print(
            f"Test chunk ID: "
            f"{test_result['ids'][0]}"
        )

        if test_result.get(
            "metadatas"
        ):

            print(
                f"Test source type: "
                f"{test_result['metadatas'][0].get('source_type')}"
            )

    else:

        print(
            "ERROR: No chunks found."
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
        "CHROMADB VECTOR STORE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Find embedding files
    # --------------------------------------------------------

    files = find_embedding_files()

    # --------------------------------------------------------
    # Load all embeddings
    # --------------------------------------------------------

    chunks = load_embeddings(
        files
    )

    print(
        f"\nTotal chunks loaded: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # Create ChromaDB
    # --------------------------------------------------------

    client, collection = (
        create_collection()
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    (
        ids,
        documents,
        embeddings,
        metadatas
    ) = prepare_data(
        chunks
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_input(
        ids,
        documents,
        embeddings,
        metadatas
    )

    # --------------------------------------------------------
    # Add to ChromaDB
    # --------------------------------------------------------

    add_to_collection(
        collection,
        ids,
        documents,
        embeddings,
        metadatas
    )

    # --------------------------------------------------------
    # Validate collection
    # --------------------------------------------------------

    validate_collection(
        collection,
        len(chunks)
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )

    print(
        f"Vector store location:\n"
        f"{VECTOR_DB_DIR}"
    )

    print(
        f"\nCollection name: "
        f"{COLLECTION_NAME}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
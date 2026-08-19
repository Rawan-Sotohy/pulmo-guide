import json
import chromadb
from pathlib import Path


# ============================================================
# PULMO GUIDE
# CHROMADB VECTOR STORE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EMBEDDINGS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "core"
    / "nice-lung-cancer-diagnosis-and-management_cleaned_embeddings.json"
)

VECTOR_DB_DIR = BASE_DIR / "data" / "vector_store"

COLLECTION_NAME = "pulmo_guide"


print("=" * 70)
print("PULMO GUIDE")
print("CHROMADB VECTOR STORE")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load embeddings
# ------------------------------------------------------------

print("\nLoading embeddings...")

with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks.")


# ------------------------------------------------------------
# 2. Create ChromaDB
# ------------------------------------------------------------

print("\nInitializing ChromaDB...")

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)


# ------------------------------------------------------------
# 3. Create / get collection
# ------------------------------------------------------------

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "Pulmo Guide Core Medical Knowledge Base"
    }
)


# ------------------------------------------------------------
# 4. Prepare data
# ------------------------------------------------------------

ids = []
documents = []
embeddings = []
metadatas = []

for chunk in chunks:

    ids.append(chunk["chunk_id"])

    documents.append(chunk["text"])

    embeddings.append(chunk["embedding"])

    metadata = chunk.get("metadata", {})

    # Chroma metadata values must be simple types
    metadatas.append({
        "source_type": metadata.get("source_type", "core"),
        "section": metadata.get("section", ""),
        "page_start": metadata.get("page_start", 0),
        "page_end": metadata.get("page_end", 0),
        "word_count": metadata.get("word_count", 0),
        "citation": chunk.get("citation", "")
    })


# ------------------------------------------------------------
# 5. Add data to ChromaDB
# ------------------------------------------------------------

print("\nAdding chunks to ChromaDB...")

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


# ------------------------------------------------------------
# 6. Validation
# ------------------------------------------------------------

count = collection.count()

print("\n" + "=" * 70)
print("VECTOR STORE VALIDATION")
print("=" * 70)

print(f"Chunks stored: {count}")

if count == len(chunks):
    print("OK: All chunks stored successfully.")
else:
    print("WARNING: Stored chunk count does not match input count.")


# ------------------------------------------------------------
# 7. Test one retrieval
# ------------------------------------------------------------

print("\nTesting vector database...")

test_result = collection.get(
    limit=1,
    include=["documents", "metadatas"]
)

if test_result["ids"]:
    print("OK: ChromaDB retrieval is working.")
    print(f"Test chunk ID: {test_result['ids'][0]}")
else:
    print("ERROR: No chunks found.")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Vector store location: {VECTOR_DB_DIR}")
"""
============================================================
PULMO GUIDE
UNIFIED END-TO-END GENERATION PIPELINE
============================================================

CORE:
    Static NICE NG122
    Stored permanently in ChromaDB

PATIENT:
    Dynamic uploaded PDF
    Any filename
    Processed only when needed
    Cached by session_id + document hash
    NEVER inserted into Core ChromaDB

IMPORTANT:
    For patient-specific questions:
        PATIENT evidence has priority.
        CORE evidence is secondary context only.

FLOW:

    USER QUERY
        ↓
    SAFETY CHECK
        ↓
    SOURCE ROUTER
        ↓
    CORE / PATIENT / BOTH
        ↓
    RETRIEVAL
        ↓
    PATIENT EVIDENCE PRIORITY
        ↓
    EVIDENCE CHECK
        ↓
    CITATIONS
        ↓
    GROUNDED PROMPT
        ↓
    LLM
        ↓
    FINAL ANSWER
============================================================
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

import sys
import json
import hashlib
import shutil
import time
import uuid

import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSING_DIR = (
    PROJECT_ROOT / "scripts" / "01_processing"
)

RETRIEVAL_DIR = (
    PROJECT_ROOT / "scripts" / "02_retrieval"
)

GENERATION_DIR = (
    PROJECT_ROOT / "scripts" / "03_generation"
)

DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# PYTHON PATHS
# ============================================================

for path in [
    PROCESSING_DIR,
    RETRIEVAL_DIR,
    GENERATION_DIR,
]:
    path_string = str(path)

    if path_string not in sys.path:
        sys.path.append(path_string)


# ============================================================
# EXISTING MODULES
# ============================================================

from retrieval import hybrid_search
from generator import generate_answer
from grounded_prompt import build_grounded_prompt
from citation import build_citation
from refusal import check_refusal
from safety import safety_check


# ============================================================
# CACHE
# ============================================================

CACHE_DIR = DATA_DIR / "patient_cache"

PATIENT_CACHE_DIR = (
    CACHE_DIR / "sessions"
)

PATIENT_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

CORE_TOP_K = 5
PATIENT_TOP_K = 5

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30

EMBEDDING_MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
)

CACHE_TTL_SECONDS = 60 * 60 * 4


# ============================================================
# SOURCE TYPES
# ============================================================

SOURCE_CORE = "core"
SOURCE_PATIENT = "patient"
SOURCE_BOTH = "core+patient"


# ============================================================
# GLOBAL EMBEDDING MODEL
# ============================================================

_EMBEDDING_MODEL = None


def get_embedding_model():
    """
    Load embedding model once and reuse it.
    """

    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is None:

        print(
            "\nLoading embedding model..."
        )

        _EMBEDDING_MODEL = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print(
            "Embedding model loaded."
        )

    return _EMBEDDING_MODEL


# ============================================================
# PATIENT QUERY KEYWORDS
# ============================================================

PATIENT_KEYWORDS = {

    # English
    "my",
    "my report",
    "my result",
    "my results",
    "my test",
    "my tests",
    "my scan",
    "my biopsy",
    "my pathology",
    "my molecular",
    "my imaging",
    "my diagnosis",
    "my finding",
    "my findings",
    "my report says",
    "my result says",

    "in my report",
    "in my results",
    "according to my report",

    "what does my report",
    "what does my scan",
    "what does my test",
    "what does my biopsy",

    # Arabic
    "تقريري",
    "تقاريري",
    "نتيجتي",
    "نتائجي",
    "تحليلي",
    "تحاليل",
    "أشعتي",
    "اشعتي",
    "الأشعة",
    "الاشعة",
    "الخزعة",
    "تشخيصي",
    "نتيجة التحليل",
    "نتيجة الأشعة",
    "نتيجة الخزعة",
    "في تقريري",
    "في تحليلي",
    "في الأشعة",
    "حسب تقريري",
    "ماذا يعني تقريري",
    "ماذا تعني نتيجتي",
}


# ============================================================
# CORE QUERY KEYWORDS
# ============================================================

CORE_KEYWORDS = {

    # English
    "what is",
    "what are",
    "symptoms",
    "causes",
    "risk factors",
    "treatment",
    "management",
    "guideline",
    "screening",
    "diagnosis",
    "staging",
    "nsclc",
    "sclc",
    "lung cancer",
    "recommendation",
    "recommendations",
    "should be offered",
    "according to",
    "nice guideline",
    "nice",
    "ng122",

    # Arabic
    "أعراض",
    "اعراض",
    "أسباب",
    "اسباب",
    "علاج",
    "تشخيص",
    "مراحل",
    "سرطان الرئة",
    "التوصيات",
    "التوصية",
    "الفحص",
    "إرشادات",
    "ارشادات",
}


# ============================================================
# QUERY NORMALIZATION
# ============================================================

def normalize_query(
    query: str
) -> str:

    return " ".join(
        query.lower()
        .strip()
        .split()
    )


# ============================================================
# FILE HASH
# ============================================================

def file_hash(
    file_path: Path
) -> str:

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# SESSION ID
# ============================================================

def create_session_id() -> str:

    return uuid.uuid4().hex


# ============================================================
# PATIENT CACHE DIRECTORY
# ============================================================

def get_patient_cache_dir(
    patient_pdf: Path,
    session_id: str
) -> Path:

    document_hash = file_hash(
        patient_pdf
    )

    cache_id = (
        f"{session_id}_"
        f"{document_hash[:16]}"
    )

    cache_dir = (
        PATIENT_CACHE_DIR / cache_id
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return cache_dir


# ============================================================
# CACHE VALIDATION
# ============================================================

def cache_is_valid(
    cache_dir: Path
) -> bool:

    metadata_path = (
        cache_dir / "cache_metadata.json"
    )

    chunks_path = (
        cache_dir / "patient_chunks.json"
    )

    if not metadata_path.exists():
        return False

    if not chunks_path.exists():
        return False

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

    except Exception:

        return False

    created_at = metadata.get(
        "created_at",
        0
    )

    if not created_at:
        return False

    age = time.time() - created_at

    return age <= CACHE_TTL_SECONDS


# ============================================================
# SAVE CACHE METADATA
# ============================================================

def save_cache_metadata(
    cache_dir: Path,
    patient_pdf: Path,
    session_id: str
):

    document_hash = file_hash(
        patient_pdf
    )

    metadata = {

        "session_id":
            session_id,

        "document_name":
            patient_pdf.name,

        "document_hash":
            document_hash,

        "created_at":
            time.time(),

        "source_type":
            SOURCE_PATIENT,

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "cache_ttl_seconds":
            CACHE_TTL_SECONDS,
    }

    metadata_path = (
        cache_dir / "cache_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# SOURCE ROUTER
# ============================================================

def determine_source_mode(
    query: str,
    patient_pdf: Optional[Path]
) -> str:

    if patient_pdf is None:
        return SOURCE_CORE

    normalized = normalize_query(query)

    patient_related = any(
        keyword in normalized
        for keyword in PATIENT_KEYWORDS
    )

    if patient_related:
        return SOURCE_BOTH

    core_related = any(
        keyword in normalized
        for keyword in CORE_KEYWORDS
    )

    if core_related:
        return SOURCE_CORE

    return SOURCE_BOTH


# ============================================================
# CORE RETRIEVAL
# ============================================================

def retrieve_core(
    query: str
) -> List[Dict[str, Any]]:

    results = hybrid_search(
        query=query,
        final_top_k=CORE_TOP_K
    )

    normalized_results = []

    for result in results:

        result = dict(result)

        metadata = dict(
            result.get("metadata") or {}
        )

        metadata["source_type"] = SOURCE_CORE

        result["metadata"] = metadata
        result["source_type"] = SOURCE_CORE

        normalized_results.append(
            result
        )

    return normalized_results


# ============================================================
# PATIENT PROCESSING
# ============================================================

def process_patient_report(
    patient_pdf: Path,
    session_id: str
) -> Dict[str, Any]:

    patient_pdf = Path(patient_pdf)

    if not patient_pdf.exists():

        raise FileNotFoundError(
            "Patient report not found: "
            f"{patient_pdf}"
        )

    cache_dir = get_patient_cache_dir(
        patient_pdf,
        session_id
    )

    chunks_path = (
        cache_dir / "patient_chunks.json"
    )

    # ========================================================
    # CACHE HIT
    # ========================================================

    if cache_is_valid(cache_dir):

        print(
            "\nPatient cache HIT."
        )

        with open(
            chunks_path,
            "r",
            encoding="utf-8"
        ) as file:

            chunks = json.load(file)

        return {
            "session_id": session_id,
            "cache_hit": True,
            "cache_dir": cache_dir,
            "chunks": chunks,
        }

    # ========================================================
    # CACHE MISS
    # ========================================================

    print(
        "\nPatient cache MISS."
    )

    print(
        "Processing uploaded patient report..."
    )

    from ingest import parse_patient
    from cleaning import clean_patient
    from section_detection import process_patient
    from chunking import process_document

    # ========================================================
    # 1. INGESTION
    # ========================================================

    print(
        "\n1. Patient ingestion..."
    )

    elements = parse_patient(
        patient_pdf
    )

    if not elements:

        raise ValueError(
            "Patient PDF produced no "
            "parsed elements."
        )

    print(
        f"   Parsed elements: {len(elements)}"
    )

    # ========================================================
    # 2. CLEANING
    # ========================================================

    print(
        "\n2. Patient cleaning..."
    )

    cleaned_elements = clean_patient(
        elements
    )

    if not cleaned_elements:

        raise ValueError(
            "Patient cleaning produced "
            "no elements."
        )

    print(
        f"   Cleaned elements: "
        f"{len(cleaned_elements)}"
    )

    # ========================================================
    # 3. SECTION DETECTION
    # ========================================================

    print(
        "\n3. Patient section detection..."
    )

    sectioned_elements = process_patient(
        cleaned_elements
    )

    if not sectioned_elements:

        raise ValueError(
            "Patient section detection "
            "produced no elements."
        )

    print(
        f"   Sectioned elements: "
        f"{len(sectioned_elements)}"
    )

    # ========================================================
    # 4. SEMANTIC MODEL
    # ========================================================

    print(
        "\n4. Loading semantic model..."
    )

    model = get_embedding_model()

    # ========================================================
    # 5. CHUNKING
    # ========================================================

    print(
        "\n5. Patient semantic chunking..."
    )

    chunks = process_document(
        sectioned_elements,
        SOURCE_PATIENT,
        model=model,
        session_id=session_id
    )

    if not chunks:

        raise ValueError(
            "Patient chunking produced "
            "no chunks."
        )

    print(
        f"   Created chunks: {len(chunks)}"
    )

    # ========================================================
    # 6. NORMALIZE METADATA
    # ========================================================

    normalized_chunks = []

    for chunk in chunks:

        chunk = dict(chunk)

        metadata = dict(
            chunk.get("metadata") or {}
        )

        metadata["source_type"] = SOURCE_PATIENT
        metadata["session_id"] = session_id

        metadata.setdefault(
            "document_name",
            patient_pdf.name
        )

        chunk["metadata"] = metadata
        chunk["source_type"] = SOURCE_PATIENT

        normalized_chunks.append(chunk)

    chunks = normalized_chunks

    # ========================================================
    # 7. SAVE CACHE
    # ========================================================

    print(
        "\n6. Saving patient cache..."
    )

    with open(
        chunks_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    save_cache_metadata(
        cache_dir,
        patient_pdf,
        session_id
    )

    print(
        "Patient processing completed."
    )

    return {
        "session_id": session_id,
        "cache_hit": False,
        "cache_dir": cache_dir,
        "chunks": chunks,
    }


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(
    scores
) -> np.ndarray:

    scores = np.asarray(
        scores,
        dtype=float
    )

    if scores.size == 0:
        return scores

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.ones_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ============================================================
# PATIENT HYBRID RETRIEVAL
# ============================================================

def patient_hybrid_search(
    query: str,
    chunks: List[Dict[str, Any]],
    final_top_k: int = PATIENT_TOP_K
) -> List[Dict[str, Any]]:

    if not chunks:
        return []

    # ========================================================
    # DOCUMENT TEXTS
    # ========================================================

    texts = [
        chunk.get("text", "")
        for chunk in chunks
    ]

    valid_indices = [
        index
        for index, text in enumerate(texts)
        if text and text.strip()
    ]

    if not valid_indices:
        return []

    texts = [
        texts[index]
        for index in valid_indices
    ]

    valid_chunks = [
        chunks[index]
        for index in valid_indices
    ]

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    model = get_embedding_model()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    document_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    semantic_scores = np.dot(
        document_embeddings,
        query_embedding
    )

    # ========================================================
    # BM25
    # ========================================================

    tokenized_documents = [
        text.lower().split()
        for text in texts
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    query_tokens = (
        query.lower().split()
    )

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    # ========================================================
    # NORMALIZE
    # ========================================================

    semantic_normalized = normalize_scores(
        semantic_scores
    )

    bm25_normalized = normalize_scores(
        bm25_scores
    )

    # ========================================================
    # HYBRID
    # ========================================================

    hybrid_scores = (
        SEMANTIC_WEIGHT * semantic_normalized
        +
        BM25_WEIGHT * bm25_normalized
    )

    # ========================================================
    # TOP K
    # ========================================================

    top_k = min(
        final_top_k,
        len(valid_chunks)
    )

    indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        indices,
        start=1
    ):

        chunk = dict(
            valid_chunks[index]
        )

        metadata = dict(
            chunk.get("metadata") or {}
        )

        metadata["source_type"] = SOURCE_PATIENT

        chunk["metadata"] = metadata
        chunk["source_type"] = SOURCE_PATIENT

        results.append({

            "hybrid_rank": rank,

            "chunk_id": chunk.get(
                "chunk_id",
                f"patient_{index}"
            ),

            "text": chunk.get(
                "text",
                ""
            ),

            "metadata": metadata,

            "semantic_score": float(
                semantic_scores[index]
            ),

            "semantic_normalized": float(
                semantic_normalized[index]
            ),

            "bm25_score": float(
                bm25_scores[index]
            ),

            "bm25_normalized": float(
                bm25_normalized[index]
            ),

            "hybrid_score": float(
                hybrid_scores[index]
            ),
        })

    return results


# ============================================================
# PATIENT RETRIEVAL
# ============================================================

def retrieve_patient(
    query: str,
    patient_pdf: Path,
    session_id: str
) -> List[Dict[str, Any]]:

    patient_data = process_patient_report(
        patient_pdf,
        session_id
    )

    chunks = patient_data["chunks"]

    if not chunks:
        return []

    results = patient_hybrid_search(
        query=query,
        chunks=chunks,
        final_top_k=PATIENT_TOP_K
    )

    normalized_results = []

    for result in results:

        result = dict(result)

        metadata = dict(
            result.get("metadata") or {}
        )

        metadata["source_type"] = SOURCE_PATIENT
        metadata["session_id"] = session_id

        metadata.setdefault(
            "document_name",
            patient_pdf.name
        )

        result["metadata"] = metadata
        result["source_type"] = SOURCE_PATIENT

        normalized_results.append(result)

    return normalized_results


# ============================================================
# PATIENT EVIDENCE PRIORITY
# ============================================================

def build_generation_results(
    source_mode: str,
    core_results: List[Dict[str, Any]],
    patient_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    IMPORTANT:

    Patient-specific queries:
        Patient evidence comes FIRST.

    Core evidence is kept only as secondary
    medical context.

    This prevents the LLM from answering a
    personal question using NICE instead of
    the uploaded report.
    """

    if source_mode == SOURCE_BOTH:

        return (
            patient_results
            +
            core_results
        )

    if source_mode == SOURCE_PATIENT:

        return patient_results

    return core_results


# ============================================================
# COMBINED RETRIEVAL
# ============================================================

def retrieve_evidence(
    query: str,
    patient_pdf: Optional[Path] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:

    source_mode = determine_source_mode(
        query,
        patient_pdf
    )

    core_results = []
    patient_results = []

    # ========================================================
    # CORE
    # ========================================================

    if source_mode in (
        SOURCE_CORE,
        SOURCE_BOTH
    ):

        print(
            "\nRetrieving from CORE..."
        )

        core_results = retrieve_core(
            query
        )

    # ========================================================
    # PATIENT
    # ========================================================

    if (
        source_mode == SOURCE_BOTH
        and patient_pdf is not None
    ):

        if session_id is None:

            raise ValueError(
                "session_id is required "
                "for Patient retrieval."
            )

        print(
            "\nRetrieving from PATIENT..."
        )

        patient_results = retrieve_patient(
            query=query,
            patient_pdf=patient_pdf,
            session_id=session_id
        )

    # ========================================================
    # GENERATION ORDER
    # ========================================================

    generation_results = build_generation_results(
        source_mode=source_mode,
        core_results=core_results,
        patient_results=patient_results
    )

    # ========================================================
    # RAW COMBINED RESULTS
    # ========================================================

    combined_results = (
        core_results
        +
        patient_results
    )

    return {

        "source_mode":
            source_mode,

        "core_results":
            core_results,

        "patient_results":
            patient_results,

        "combined_results":
            combined_results,

        "generation_results":
            generation_results,
    }


# ============================================================
# CITATIONS
# ============================================================

def attach_citations(
    retrieved_results: List[Dict[str, Any]]
) -> List[str]:

    citations = []

    for result in retrieved_results:

        metadata = (
            result.get("metadata")
            or {}
        )

        citation = build_citation(
            metadata
        )

        result["citation"] = citation

        if citation not in citations:
            citations.append(citation)

    return citations


# ============================================================
# CLEAN PATIENT SESSION
# ============================================================

def cleanup_patient_session(
    session_id: str
):

    if not session_id:
        return

    for cache_dir in PATIENT_CACHE_DIR.glob(
        f"{session_id}_*"
    ):

        if cache_dir.exists():

            shutil.rmtree(
                cache_dir,
                ignore_errors=True
            )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    query: str,
    patient_pdf: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    patient_path = None

    # ========================================================
    # PATIENT PDF
    # ========================================================

    if patient_pdf:

        patient_path = Path(
            patient_pdf
        )

        if not patient_path.exists():

            raise FileNotFoundError(
                "Patient report not found: "
                f"{patient_path}"
            )

        if session_id is None:
            session_id = create_session_id()

    # ========================================================
    # 1. SAFETY
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "1. SAFETY CHECK"
    )

    print(
        "=" * 75
    )

    safety_result = safety_check(
        query=query,
        patient_pdf=(
            patient_path is not None
        )
    )

    if not safety_result[
        "retrieval_allowed"
    ]:

        return {

            "status": "refused",
            "stage": "safety",
            "query": query,
            "session_id": session_id,
            "source_mode": None,
            "safety": safety_result,
            "evidence": None,
            "citations": [],
            "retrieved_results": [],
            "answer": safety_result["message"],
        }

    # ========================================================
    # 2. SOURCE ROUTING + RETRIEVAL
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2. SOURCE ROUTING + RETRIEVAL"
    )

    print(
        "=" * 75
    )

    retrieval_data = retrieve_evidence(
        query=query,
        patient_pdf=patient_path,
        session_id=session_id,
    )

    source_mode = retrieval_data[
        "source_mode"
    ]

    core_results = retrieval_data[
        "core_results"
    ]

    patient_results = retrieval_data[
        "patient_results"
    ]

    # ========================================================
    # IMPORTANT:
    # Use patient-first ordering for generation.
    # ========================================================

    generation_results = retrieval_data[
        "generation_results"
    ]

    print(
        f"\nSource mode: {source_mode}"
    )

    print(
        f"Core results: {len(core_results)}"
    )

    print(
        f"Patient results: {len(patient_results)}"
    )

    # ========================================================
    # SHOW PATIENT EVIDENCE
    # ========================================================

    if patient_results:

        print(
            "\n"
            + "-" * 75
        )

        print(
            "PATIENT EVIDENCE USED FOR GENERATION"
        )

        print(
            "-" * 75
        )

        for index, item in enumerate(
            patient_results,
            start=1
        ):

            metadata = (
                item.get("metadata")
                or {}
            )

            print(
                f"\n[{index}] "
                f"{item.get('chunk_id')}"
            )

            print(
                "Section:",
                metadata.get(
                    "section",
                    ""
                )
            )

            print(
                "Page:",
                metadata.get(
                    "page_start",
                    metadata.get(
                        "page",
                        ""
                    )
                )
            )

            print(
                "Score:",
                item.get(
                    "hybrid_score",
                    ""
                )
            )

            print(
                "Text:"
            )

            print(
                item.get(
                    "text",
                    ""
                )
            )

    # ========================================================
    # 3. NO EVIDENCE
    # ========================================================

    if not generation_results:

        return {

            "status": "refused",
            "stage": "retrieval",
            "query": query,
            "session_id": session_id,
            "source_mode": source_mode,
            "safety": safety_result,
            "evidence": None,
            "citations": [],
            "core_results": core_results,
            "patient_results": patient_results,
            "retrieved_results": [],
            "answer": (
                "I couldn't find enough relevant "
                "evidence to answer this question "
                "confidently."
            ),
        }

    # ========================================================
    # 4. EVIDENCE CHECK
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "3. EVIDENCE CHECK"
    )

    print(
        "=" * 75
    )

    evidence_result = check_refusal(
        query=query,
        retrieved_results=generation_results,
    )

    if (
        evidence_result["decision"]
        == "insufficient"
    ):

        return {

            "status": "refused",
            "stage": "evidence",
            "query": query,
            "session_id": session_id,
            "source_mode": source_mode,
            "safety": safety_result,
            "evidence": evidence_result,
            "citations": [],
            "core_results": core_results,
            "patient_results": patient_results,
            "retrieved_results": generation_results,
            "answer": (
                "I couldn't find enough relevant "
                "evidence to answer this question "
                "confidently."
            ),
        }

    # ========================================================
    # 5. CITATIONS
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "4. CITATIONS"
    )

    print(
        "=" * 75
    )

    citations = attach_citations(
        generation_results
    )

    # ========================================================
    # 6. GROUNDED PROMPT
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "5. GROUNDED PROMPT"
    )

    print(
        "=" * 75
    )

    persona = safety_result[
        "persona"
    ]

    if hasattr(
        persona,
        "value"
    ):

        persona = persona.value

    # ========================================================
    # PATIENT PRIORITY INSTRUCTION
    # ========================================================

    patient_priority_instruction = ""

    if source_mode == SOURCE_BOTH:

        patient_priority_instruction = """
IMPORTANT PATIENT-SPECIFIC RULE:

The user has uploaded a patient report and is
asking about their personal result.

PATIENT evidence has the highest priority.

If the patient's uploaded report contains the
requested measurement, value, result, finding,
or interpretation, answer from the PATIENT
evidence.

DO NOT replace the patient's actual value with
a general NICE guideline definition.

CORE/NICE evidence may be used only as secondary
context or explanation.

If the requested patient value is not present
in the patient evidence, explicitly say that the
specific value was not found.

Never invent or estimate a patient value.
"""

    # ========================================================
    # BUILD GROUNDED PROMPT
    # ========================================================

    grounded_prompt = build_grounded_prompt(
        query=query,
        retrieved_results=generation_results,
        persona=persona,
        evidence_level=(
            evidence_result[
                "evidence_level"
            ]
        ),
    )

    # Add patient-priority instruction AFTER
    # the generated grounded prompt so it cannot
    # be ignored by the generic prompt.

    if patient_priority_instruction:

        grounded_prompt = (
            grounded_prompt
            + "\n\n"
            + patient_priority_instruction
        )

    # ========================================================
    # 7. GENERATION
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "6. LLM GENERATION"
    )

    print(
        "=" * 75
    )

    generation_result = generate_answer(
        grounded_prompt=grounded_prompt
    )

    # ========================================================
    # 8. FINAL RESULT
    # ========================================================

    return {

        "status": "success",

        "stage": "generation",

        "query": query,

        "session_id": session_id,

        "source_mode": source_mode,

        "safety": safety_result,

        "evidence": evidence_result,

        "core_results": core_results,

        "patient_results": patient_results,

        # Patient-first results used by LLM
        "generation_results":
            generation_results,

        "retrieved_results":
            generation_results,

        "citations": citations,

        "grounded_prompt":
            grounded_prompt,

        "answer":
            generation_result[
                "answer"
            ],

        "generation":
            generation_result,
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_pipeline_result(
    result: Dict[str, Any]
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PULMO GUIDE - PIPELINE RESULT"
    )

    print(
        "=" * 75
    )

    print(
        "\nStatus:",
        result.get("status")
    )

    print(
        "Stage:",
        result.get("stage")
    )

    print(
        "Source Mode:",
        result.get("source_mode")
    )

    print(
        "Session ID:",
        result.get("session_id")
    )

    # ========================================================
    # CORE
    # ========================================================

    print(
        "\n"
        + "-" * 75
    )

    print(
        "CORE RESULTS"
    )

    print(
        "-" * 75
    )

    core_results = result.get(
        "core_results",
        []
    )

    if not core_results:
        print("No Core results.")

    for item in core_results:

        metadata = (
            item.get("metadata")
            or {}
        )

        print(
            item.get("chunk_id"),
            "|",
            metadata.get(
                "section",
                ""
            ),
            "|",
            metadata.get(
                "page_start",
                metadata.get(
                    "page",
                    ""
                )
            ),
            "| score:",
            item.get(
                "hybrid_score",
                ""
            )
        )

    # ========================================================
    # PATIENT
    # ========================================================

    print(
        "\n"
        + "-" * 75
    )

    print(
        "PATIENT RESULTS"
    )

    print(
        "-" * 75
    )

    patient_results = result.get(
        "patient_results",
        []
    )

    if not patient_results:
        print("No Patient results.")

    for item in patient_results:

        metadata = (
            item.get("metadata")
            or {}
        )

        print(
            item.get("chunk_id"),
            "|",
            metadata.get(
                "section",
                ""
            ),
            "|",
            metadata.get(
                "page_start",
                metadata.get(
                    "page",
                    ""
                )
            ),
            "| score:",
            item.get(
                "hybrid_score",
                ""
            )
        )

    # ========================================================
    # CITATIONS
    # ========================================================

    print(
        "\n"
        + "-" * 75
    )

    print(
        "CITATIONS"
    )

    print(
        "-" * 75
    )

    citations = result.get(
        "citations",
        []
    )

    if not citations:
        print("No citations.")

    for citation in citations:
        print(citation)

    # ========================================================
    # ANSWER
    # ========================================================

    print(
        "\n"
        + "-" * 75
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "-" * 75
    )

    print(
        result.get(
            "answer",
            ""
        )
    )

    print(
        "\n"
        + "=" * 75
    )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 75
    )

    print(
        "PULMO GUIDE - END-TO-END PIPELINE TEST"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # TEST 1 - CORE
    # ========================================================

    test_query = (
        "What imaging should be offered to people "
        "with stage 3 NSCLC who are having treatment "
        "with curative intent?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 1 - CORE"
    )

    print(
        "=" * 75
    )

    print(
        "\nQuery:"
    )

    print(
        test_query
    )

    try:

        result = run_pipeline(
            query=test_query,
            patient_pdf=None
        )

        print_pipeline_result(
            result
        )

    except Exception as error:

        print(
            "\nCORE PIPELINE ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            str(error)
        )

    # ========================================================
    # TEST 2 - PATIENT
    # ========================================================

    patient_pdf = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "patient"
        / "pulmonary_function_report.pdf"
    )

    patient_query = (
        "What is my FEV1?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 2 - PATIENT"
    )

    print(
        "=" * 75
    )

    print(
        "\nPatient PDF:"
    )

    print(
        patient_pdf
    )

    print(
        "\nPatient Query:"
    )

    print(
        patient_query
    )

    if patient_pdf.exists():

        try:

            result = run_pipeline(
                query=patient_query,
                patient_pdf=str(
                    patient_pdf
                ),
                session_id=(
                    "test_patient_001"
                )
            )

            print_pipeline_result(
                result
            )

        except Exception as error:

            print(
                "\nPATIENT PIPELINE ERROR:"
            )

            print(
                type(error).__name__,
                ":",
                str(error)
            )

    else:

        print(
            "\nPatient test skipped."
        )

        print(
            "PDF not found:"
        )

        print(
            patient_pdf
        )

    # ========================================================
    # TEST INFORMATION
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        """
Patient PDF filename can be anything.

The pipeline identifies the document using:

    session_id
        +
    document hash

Patient data is cached temporarily.

Patient data is NEVER inserted
into Core ChromaDB.

Core:
    NICE NG122
    ChromaDB

Patient:
    Uploaded PDF
    Temporary cache
    Direct hybrid retrieval

Patient retrieval:
    Semantic 70%
    BM25 30%

For patient-specific questions:
    PATIENT evidence has priority.
"""
    )

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

PROCESSING_DIR = PROJECT_ROOT / "scripts" / "01_processing"
RETRIEVAL_DIR = PROJECT_ROOT / "scripts" / "02_retrieval"
GENERATION_DIR = PROJECT_ROOT / "scripts" / "03_generation"

DATA_DIR = PROJECT_ROOT / "data"

CACHE_DIR = DATA_DIR / "patient_cache"
PATIENT_CACHE_DIR = CACHE_DIR / "sessions"

PATIENT_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PYTHON PATHS
# ============================================================

sys.path.append(str(PROCESSING_DIR))
sys.path.append(str(RETRIEVAL_DIR))
sys.path.append(str(GENERATION_DIR))


# ============================================================
# EXISTING MODULES
# ============================================================

from safety import safety_check
from refusal import check_refusal
from citation import build_citation
from grounded_prompt import build_grounded_prompt
from generator import generate_answer

from retrieval import hybrid_search


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
# PATIENT QUERY KEYWORDS
# ============================================================

PATIENT_KEYWORDS = {

    "my",
    "me",
    "patient",
    "my report",
    "my results",
    "my test",
    "my scan",
    "my biopsy",
    "my pathology",
    "my molecular",
    "my imaging",
    "my diagnosis",
    "my findings",
    "my report says",

    "المريض",
    "تقريبي",
    "تقاريري",
    "تحاليل",
    "تحليلي",
    "اشعتي",
    "الأشعة",
    "الخزعة",
    "نتيجتي",
    "نتائجي",
    "تشخيصي",
    "تقريري",
}


# ============================================================
# CORE QUERY KEYWORDS
# ============================================================

CORE_KEYWORDS = {

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
    "should be offered",
    "according to",

    "اعراض",
    "أعراض",
    "علاج",
    "تشخيص",
    "مراحل",
    "سرطان الرئة",
    "التوصيات",
}


# ============================================================
# QUERY NORMALIZATION
# ============================================================

def normalize_query(query: str) -> str:

    return " ".join(
        query.lower().strip().split()
    )


# ============================================================
# FILE HASH
# ============================================================

def file_hash(file_path: Path) -> str:

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
        f"{session_id}_{document_hash[:16]}"
    )

    cache_dir = (
        PATIENT_CACHE_DIR /
        cache_id
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
        cache_dir /
        "cache_metadata.json"
    )

    if not metadata_path.exists():
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

    age = (
        time.time()
        - created_at
    )

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
    }

    metadata_path = (
        cache_dir /
        "cache_metadata.json"
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

    normalized = normalize_query(
        query
    )

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

        metadata["source_type"] = (
            SOURCE_CORE
        )

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

    patient_pdf = Path(
        patient_pdf
    )

    if not patient_pdf.exists():

        raise FileNotFoundError(
            f"Patient report not found: "
            f"{patient_pdf}"
        )

    cache_dir = get_patient_cache_dir(
        patient_pdf,
        session_id
    )

    chunks_path = (
        cache_dir /
        "patient_chunks.json"
    )

    # --------------------------------------------------------
    # CACHE HIT
    # --------------------------------------------------------

    if (
        cache_is_valid(cache_dir)
        and chunks_path.exists()
    ):

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

            "session_id":
                session_id,

            "cache_hit":
                True,

            "cache_dir":
                cache_dir,

            "chunks":
                chunks,
        }

    # --------------------------------------------------------
    # CACHE MISS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 1. INGESTION
    # --------------------------------------------------------

    print(
        "1. Patient ingestion..."
    )

    elements = parse_patient(
        patient_pdf
    )

    # --------------------------------------------------------
    # 2. CLEANING
    # --------------------------------------------------------

    print(
        "2. Patient cleaning..."
    )

    cleaned_elements = clean_patient(
        elements
    )

    # --------------------------------------------------------
    # 3. SECTION DETECTION
    # --------------------------------------------------------

    print(
        "3. Patient section detection..."
    )

    sectioned_elements = process_patient(
        cleaned_elements
    )

    # --------------------------------------------------------
    # 4. CHUNKING
    # --------------------------------------------------------

    print(
        "4. Patient chunking..."
    )

    chunks = process_document(
        sectioned_elements,
        SOURCE_PATIENT,
        model=None,
        session_id=session_id
    )

    # --------------------------------------------------------
    # 5. NORMALIZE METADATA
    # --------------------------------------------------------

    normalized_chunks = []

    for chunk in chunks:

        chunk = dict(chunk)

        metadata = dict(
            chunk.get("metadata") or {}
        )

        metadata["source_type"] = (
            SOURCE_PATIENT
        )

        metadata["session_id"] = (
            session_id
        )

        metadata.setdefault(
            "document_name",
            patient_pdf.name
        )

        chunk["metadata"] = metadata

        chunk["source_type"] = (
            SOURCE_PATIENT
        )

        normalized_chunks.append(
            chunk
        )

    chunks = normalized_chunks

    # --------------------------------------------------------
    # 6. SAVE CACHE
    # --------------------------------------------------------

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

        "session_id":
            session_id,

        "cache_hit":
            False,

        "cache_dir":
            cache_dir,

        "chunks":
            chunks,
    }


# ============================================================
# PATIENT HYBRID RETRIEVAL
# ============================================================

def patient_hybrid_search(
    query: str,
    chunks: List[Dict[str, Any]],
    final_top_k: int = PATIENT_TOP_K
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval directly over temporary
    patient chunks.

    No ChromaDB.

    Semantic 70%
    BM25 30%
    """

    if not chunks:
        return []

    texts = []

    for chunk in chunks:

        text = chunk.get(
            "text",
            ""
        )

        texts.append(
            text
        )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    document_embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    semantic_scores = np.dot(
        document_embeddings,
        query_embedding
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    def normalize(scores):

        scores = np.asarray(
            scores,
            dtype=float
        )

        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:

            return np.zeros_like(
                scores
            )

        return (
            (scores - minimum)
            /
            (maximum - minimum)
        )

    semantic_normalized = normalize(
        semantic_scores
    )

    bm25_normalized = normalize(
        bm25_scores
    )

    # --------------------------------------------------------
    # Hybrid score
    # --------------------------------------------------------

    hybrid_scores = (

        SEMANTIC_WEIGHT
        * semantic_normalized

        +

        BM25_WEIGHT
        * bm25_normalized
    )

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    indices = np.argsort(
        hybrid_scores
    )[::-1][
        :final_top_k
    ]

    results = []

    for rank, index in enumerate(
        indices,
        start=1
    ):

        chunk = dict(
            chunks[index]
        )

        metadata = dict(
            chunk.get("metadata") or {}
        )

        metadata["source_type"] = (
            SOURCE_PATIENT
        )

        chunk["metadata"] = metadata
        chunk["source_type"] = SOURCE_PATIENT

        results.append({

            "hybrid_rank":
                rank,

            "chunk_id":
                chunk.get(
                    "chunk_id",
                    f"patient_{index}"
                ),

            "text":
                chunk.get(
                    "text",
                    ""
                ),

            "metadata":
                metadata,

            "semantic_score":
                float(
                    semantic_scores[index]
                ),

            "semantic_normalized":
                float(
                    semantic_normalized[index]
                ),

            "bm25_score":
                float(
                    bm25_scores[index]
                ),

            "bm25_normalized":
                float(
                    bm25_normalized[index]
                ),

            "hybrid_score":
                float(
                    hybrid_scores[index]
                )
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

    chunks = patient_data[
        "chunks"
    ]

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

        metadata["source_type"] = (
            SOURCE_PATIENT
        )

        metadata["session_id"] = (
            session_id
        )

        metadata.setdefault(
            "document_name",
            patient_pdf.name
        )

        result["metadata"] = metadata
        result["source_type"] = SOURCE_PATIENT

        normalized_results.append(
            result
        )

    return normalized_results


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

    # --------------------------------------------------------
    # CORE
    # --------------------------------------------------------

    if source_mode in (
        SOURCE_CORE,
        SOURCE_BOTH
    ):

        core_results = retrieve_core(
            query
        )

    # --------------------------------------------------------
    # PATIENT
    # --------------------------------------------------------

    if (
        source_mode == SOURCE_BOTH
        and patient_pdf is not None
    ):

        if session_id is None:

            raise ValueError(
                "session_id is required "
                "for Patient retrieval."
            )

        patient_results = retrieve_patient(
            query=query,
            patient_pdf=patient_pdf,
            session_id=session_id
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

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
    }


# ============================================================
# CITATIONS
# ============================================================

def attach_citations(
    retrieved_results:
        List[Dict[str, Any]]
):

    citations = []

    for result in retrieved_results:

        metadata = (
            result.get("metadata")
            or {}
        )

        citation = build_citation(
            metadata
        )

        result["citation"] = (
            citation
        )

        if citation not in citations:

            citations.append(
                citation
            )

    return citations


# ============================================================
# CLEAN PATIENT SESSION
# ============================================================

def cleanup_patient_session(
    session_id: str
):

    if not session_id:
        return

    for cache_dir in (
        PATIENT_CACHE_DIR.glob(
            f"{session_id}_*"
        )
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

    # --------------------------------------------------------
    # Patient PDF
    # --------------------------------------------------------

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

    safety_result = safety_check(
        query
    )

    if not safety_result[
        "retrieval_allowed"
    ]:

        return {

            "status":
                "refused",

            "stage":
                "safety",

            "query":
                query,

            "session_id":
                session_id,

            "source_mode":
                None,

            "safety":
                safety_result,

            "evidence":
                None,

            "citations":
                [],

            "retrieved_results":
                [],

            "answer":
                safety_result[
                    "message"
                ],
        }

    # ========================================================
    # 2. RETRIEVAL
    # ========================================================

    retrieval_data = retrieve_evidence(

        query=query,

        patient_pdf=patient_path,

        session_id=session_id,
    )

    retrieved_results = (
        retrieval_data[
            "combined_results"
        ]
    )

    # ========================================================
    # 3. NO EVIDENCE
    # ========================================================

    if not retrieved_results:

        return {

            "status":
                "refused",

            "stage":
                "retrieval",

            "query":
                query,

            "session_id":
                session_id,

            "source_mode":
                retrieval_data[
                    "source_mode"
                ],

            "safety":
                safety_result,

            "evidence":
                None,

            "citations":
                [],

            "retrieved_results":
                [],

            "answer":
                (
                    "I couldn't find enough "
                    "relevant evidence to answer "
                    "this question confidently."
                ),
        }

    # ========================================================
    # 4. EVIDENCE CHECK
    # ========================================================

    evidence_result = check_refusal(

        query=query,

        retrieved_results=
            retrieved_results,
    )

    if (
        evidence_result["decision"]
        == "insufficient"
    ):

        return {

            "status":
                "refused",

            "stage":
                "evidence",

            "query":
                query,

            "session_id":
                session_id,

            "source_mode":
                retrieval_data[
                    "source_mode"
                ],

            "safety":
                safety_result,

            "evidence":
                evidence_result,

            "citations":
                [],

            "retrieved_results":
                retrieved_results,

            "answer":
                (
                    "I couldn't find enough "
                    "relevant evidence to answer "
                    "this question confidently."
                ),
        }

    # ========================================================
    # 5. CITATIONS
    # ========================================================

    citations = attach_citations(
        retrieved_results
    )

    # ========================================================
    # 6. GROUNDED PROMPT
    # ========================================================

    grounded_prompt = build_grounded_prompt(

        query=query,

        retrieved_results=
            retrieved_results,

        persona=safety_result[
            "persona"
        ].value,

        evidence_level=
            evidence_result[
                "evidence_level"
            ],
    )

    # ========================================================
    # 7. GENERATION
    # ========================================================

    generation_result = generate_answer(

        grounded_prompt=
            grounded_prompt
    )

    # ========================================================
    # 8. FINAL RESULT
    # ========================================================

    return {

        "status":
            "success",

        "stage":
            "generation",

        "query":
            query,

        "session_id":
            session_id,

        "source_mode":
            retrieval_data[
                "source_mode"
            ],

        "safety":
            safety_result,

        "evidence":
            evidence_result,

        "core_results":
            retrieval_data[
                "core_results"
            ],

        "patient_results":
            retrieval_data[
                "patient_results"
            ],

        "retrieved_results":
            retrieved_results,

        "citations":
            citations,

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
        "\nSource Mode:",
        result.get(
            "source_mode"
        )
    )

    print(
        "Session ID:",
        result.get(
            "session_id"
        )
    )

    # --------------------------------------------------------
    # CORE
    # --------------------------------------------------------

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

    for item in result.get(
        "core_results",
        []
    ):

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        print(
            item.get(
                "chunk_id"
            ),
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
            )
        )

    # --------------------------------------------------------
    # PATIENT
    # --------------------------------------------------------

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

    for item in result.get(
        "patient_results",
        []
    ):

        metadata = (
            item.get(
                "metadata"
            )
            or {}
        )

        print(
            item.get(
                "chunk_id"
            ),
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
            )
        )

    # --------------------------------------------------------
    # CITATIONS
    # --------------------------------------------------------

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

    for citation in result.get(
        "citations",
        []
    ):

        print(
            citation
        )

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CORE TEST
    # --------------------------------------------------------

    test_query = (
        "What imaging should be offered to people "
        "with stage 3 NSCLC who are having treatment "
        "with curative intent?"
    )

    print(
        "\nTEST - CORE"
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
            "\nPIPELINE ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            str(error)
        )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "PATIENT TEST"
    )

    print(
        "=" * 75
    )

    print(
        """
Example:

result = run_pipeline(
    query="What is my FEV1?",
    patient_pdf="ANY_FILENAME.pdf",
    session_id="session_123"
)

print_pipeline_result(result)

The patient PDF filename does not matter.
Patient data is cached temporarily.
Patient data is NOT inserted into Core ChromaDB.
"""
    )
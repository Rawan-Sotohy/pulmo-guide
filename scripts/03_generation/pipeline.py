"""
============================================================
PULMO GUIDE
UNIFIED END-TO-END GENERATION PIPELINE
============================================================

DAY 3 PIPELINE

CORE:
    Static NICE NG122
    Stored permanently in ChromaDB

PATIENT:
    Dynamic uploaded PDF
    Processed only when needed
    Cached by session_id + document hash

PIPELINE:
    1. Safety
    2. Scope
    3. Source Routing
    4. Retrieval
    5. Evidence Check
    6. Citations
    7. Grounded Prompt
    8. LLM Generation
    9. Grounded Fallback if LLM quota is exhausted

IMPORTANT:
    - Patient evidence has priority for patient-specific facts.
    - Patient data NEVER enters Core ChromaDB.
    - Out-of-scope questions are refused before generation.
    - Safety refusal is a valid refusal.
    - Evidence refusal is a valid refusal.
    - LLM quota exhaustion does not produce an empty answer.
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
PATIENT_CACHE_DIR = CACHE_DIR / "sessions"

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

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

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

    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is None:

        print("\nLoading embedding model...")

        _EMBEDDING_MODEL = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print("Embedding model loaded.")

    return _EMBEDDING_MODEL


# ============================================================
# PATIENT QUERY KEYWORDS
# ============================================================

PATIENT_KEYWORDS = {

    "my",
    "mine",
    "my report",
    "my result",
    "my results",
    "my test",
    "my tests",
    "my scan",
    "my biopsy",
    "my pathology",
    "my imaging",
    "my diagnosis",
    "my finding",
    "my findings",

    "in my report",
    "in my results",
    "according to my report",

    "what does my report",
    "what does my scan",
    "what does my test",
    "what does my biopsy",

    "this result",
    "this finding",
    "this report",
    "these results",
    "these findings",
    "this test",
    "this scan",

    "what does this mean",
    "what does this result mean",
    "what does this finding mean",
    "what do these results mean",
    "what do these findings mean",

    "is this normal",
    "is this result normal",
    "is this finding normal",

    "should i be concerned",
    "is this concerning",

    "fev1",
    "fvc",
    "fev1/fvc",
    "tlco",
    "kco",
    "pef",

    "tumor size",
    "lesion size",
    "lymph node",
    "mutation",
    "biomarker",
    "egfr",
    "alk",
    "ros1",
    "pd-l1",

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
    "النتيجة دي",
    "النتيجة هذه",
    "النتيجة",
    "التحليل ده",
    "التحليل هذا",
    "التقرير ده",
    "التقرير هذا",
    "ده معناه ايه",
    "دي معناها ايه",
    "يعني ايه",
    "هل ده طبيعي",
    "هل دي طبيعية",
}


# ============================================================
# CORE QUERY KEYWORDS
# ============================================================

CORE_KEYWORDS = {

    "what is",
    "what are",
    "what does",
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
    "lung",
    "recommendation",
    "recommendations",
    "should be offered",
    "according to",
    "nice guideline",
    "nice",
    "ng122",

    "أعراض",
    "اعراض",
    "أسباب",
    "اسباب",
    "علاج",
    "تشخيص",
    "مراحل",
    "سرطان الرئة",
    "الرئة",
    "التوصيات",
    "التوصية",
    "الفحص",
    "إرشادات",
    "ارشادات",
}


# ============================================================
# EXPLICIT OUT-OF-SCOPE TERMS
# ============================================================

OUT_OF_SCOPE_TERMS = {

    "pancreatic cancer",
    "pancreas cancer",
    "breast cancer",
    "colon cancer",
    "colorectal cancer",
    "prostate cancer",
    "liver cancer",
    "kidney cancer",
    "brain cancer",
    "skin cancer",
    "cervical cancer",
    "ovarian cancer",
    "stomach cancer",
    "thyroid cancer",

    "pancreatic tumor",
    "breast tumor",
    "colon tumor",
    "prostate tumor",
    "liver tumor",

    "سرطان البنكرياس",
    "سرطان الثدي",
    "سرطان القولون",
    "سرطان البروستاتا",
    "سرطان الكبد",
    "سرطان المخ",
    "سرطان الجلد",
    "سرطان المعدة",
    "سرطان الغدة الدرقية",
}


# ============================================================
# GENERIC OUT-OF-SCOPE PHRASES
# ============================================================

OUT_OF_SCOPE_PHRASES = {

    "condition that is not covered",
    "condition not covered",
    "not covered by the indexed",
    "not covered by this guideline",
    "not covered by the guideline",
    "outside the scope of the guideline",
    "outside the scope",
    "not covered in the guideline",
    "not addressed by the guideline",
    "not addressed in the guideline",
    "unrelated condition",
    "unrelated disease",
    "condition outside",
    "disease outside",

    "حالة غير مغطاة",
    "مرض غير مغطى",
    "خارج نطاق الدليل",
    "غير مذكور في الدليل",
    "غير مغطى في الدليل",
}


# ============================================================
# QUERY NORMALIZATION
# ============================================================

def normalize_query(query: str) -> str:

    return " ".join(
        query.lower()
        .strip()
        .split()
    )


# ============================================================
# EXPLICIT OUT-OF-SCOPE CHECK
# ============================================================

def is_explicitly_out_of_scope(
    query: str
) -> bool:

    normalized = normalize_query(query)

    if any(
        term in normalized
        for term in OUT_OF_SCOPE_TERMS
    ):
        return True

    if any(
        phrase in normalized
        for phrase in OUT_OF_SCOPE_PHRASES
    ):
        return True

    return False


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

            chunk = file.read(1024 * 1024)

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

    document_hash = file_hash(patient_pdf)

    cache_id = (
        f"{session_id}_"
        f"{document_hash[:16]}"
    )

    cache_dir = PATIENT_CACHE_DIR / cache_id

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

    metadata_path = cache_dir / "cache_metadata.json"
    chunks_path = cache_dir / "patient_chunks.json"

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

    document_hash = file_hash(patient_pdf)

    metadata = {

        "session_id": session_id,

        "document_name": patient_pdf.name,

        "document_hash": document_hash,

        "created_at": time.time(),

        "source_type": SOURCE_PATIENT,

        "embedding_model": EMBEDDING_MODEL_NAME,

        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }

    metadata_path = cache_dir / "cache_metadata.json"

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

        normalized_results.append(result)

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

    chunks_path = cache_dir / "patient_chunks.json"

    if cache_is_valid(cache_dir):

        print("\nPatient cache HIT.")

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

    print("\nPatient cache MISS.")
    print("Processing uploaded patient report...")

    from ingest import parse_patient
    from cleaning import clean_patient
    from section_detection import process_patient
    from chunking import process_document

    print("\n1. Patient ingestion...")

    elements = parse_patient(patient_pdf)

    if not elements:
        raise ValueError(
            "Patient PDF produced no parsed elements."
        )

    print(f"   Parsed elements: {len(elements)}")

    print("\n2. Patient cleaning...")

    cleaned_elements = clean_patient(elements)

    if not cleaned_elements:
        raise ValueError(
            "Patient cleaning produced no elements."
        )

    print(
        f"   Cleaned elements: "
        f"{len(cleaned_elements)}"
    )

    print("\n3. Patient section detection...")

    sectioned_elements = process_patient(
        cleaned_elements
    )

    if not sectioned_elements:
        raise ValueError(
            "Patient section detection produced no elements."
        )

    print(
        f"   Sectioned elements: "
        f"{len(sectioned_elements)}"
    )

    print("\n4. Loading semantic model...")

    model = get_embedding_model()

    print("\n5. Patient semantic chunking...")

    chunks = process_document(
        sectioned_elements,
        SOURCE_PATIENT,
        model=model,
        session_id=session_id
    )

    if not chunks:
        raise ValueError(
            "Patient chunking produced no chunks."
        )

    print(f"   Created chunks: {len(chunks)}")

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

    print("\n6. Saving patient cache...")

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

    print("Patient processing completed.")

    return {
        "session_id": session_id,
        "cache_hit": False,
        "cache_dir": cache_dir,
        "chunks": chunks,
    }


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(scores) -> np.ndarray:

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

    tokenized_documents = [
        text.lower().split()
        for text in texts
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    semantic_normalized = normalize_scores(
        semantic_scores
    )

    bm25_normalized = normalize_scores(
        bm25_scores
    )

    hybrid_scores = (
        SEMANTIC_WEIGHT * semantic_normalized
        +
        BM25_WEIGHT * bm25_normalized
    )

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
# GENERATION ORDER
# ============================================================

def build_generation_results(
    source_mode: str,
    core_results: List[Dict[str, Any]],
    patient_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    if source_mode == SOURCE_BOTH:
        return patient_results + core_results

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

    if source_mode in (
        SOURCE_CORE,
        SOURCE_BOTH
    ):

        print("\nRetrieving from CORE...")

        core_results = retrieve_core(query)

    if (
        source_mode in (
            SOURCE_PATIENT,
            SOURCE_BOTH
        )
        and patient_pdf is not None
    ):

        if session_id is None:
            raise ValueError(
                "session_id is required for Patient retrieval."
            )

        print("\nRetrieving from PATIENT...")

        patient_results = retrieve_patient(
            query=query,
            patient_pdf=patient_pdf,
            session_id=session_id
        )

    generation_results = build_generation_results(
        source_mode=source_mode,
        core_results=core_results,
        patient_results=patient_results
    )

    return {

        "source_mode": source_mode,

        "core_results": core_results,

        "patient_results": patient_results,

        "combined_results":
            core_results + patient_results,

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

        metadata = result.get("metadata") or {}

        citation = build_citation(metadata)

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
# STANDARD REFUSAL RESULT
# ============================================================

def build_refusal_result(
    *,
    query: str,
    session_id: Optional[str],
    stage: str,
    message: str,
    source_mode: Optional[str] = None,
    safety: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    core_results: Optional[List] = None,
    patient_results: Optional[List] = None,
    retrieved_results: Optional[List] = None,
) -> Dict[str, Any]:

    return {

        "status": "refused",

        "stage": stage,

        "query": query,

        "session_id": session_id,

        "source_mode": source_mode,

        "safety": safety,

        "evidence": evidence,

        "core_results": core_results or [],

        "patient_results": patient_results or [],

        "retrieved_results":
            retrieved_results or [],

        "citations": [],

        "grounded_prompt": None,

        "answer": message,

        "generation": None,
    }


# ============================================================
# GROUNDED FALLBACK ANSWER
# ============================================================

def build_fallback_answer(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    citations: List[str],
    source_mode: str
) -> str:
    """
    Safe fallback when the LLM is unavailable.

    Uses only retrieved evidence.
    Never adds external medical knowledge.
    """

    if not retrieved_results:
        return (
            "I couldn't find enough relevant evidence "
            "to answer this question confidently."
        )

    # Patient-specific value question
    normalized = normalize_query(query)

    if (
        source_mode == SOURCE_BOTH
        and (
            "fev1" in normalized
            or "fvc" in normalized
            or "tlco" in normalized
            or "kco" in normalized
            or "pef" in normalized
        )
    ):

        patient_results = [
            result
            for result in retrieved_results
            if result.get("source_type") == SOURCE_PATIENT
        ]

        if patient_results:

            evidence_text = " ".join(
                result.get("text", "")
                for result in patient_results[:2]
            )

            # FEV1
            if "fev1" in normalized:

                import re

                match = re.search(
                    r"FEV1\s*(?:\(L\))?\s*[^%\n]*?(\d{2,3})%",
                    evidence_text,
                    re.IGNORECASE
                )

                if match:

                    value = match.group(1)

                    return (
                        f"According to the uploaded report, "
                        f"your FEV1 is {value}% predicted.\n\n"
                        f"This answer is based only on the "
                        f"patient report evidence retrieved "
                        f"for your question.\n\n"
                        f"Citations:\n"
                        + "\n".join(
                            f"- {citation}"
                            for citation in citations
                            if "pulmonary_function_report" in citation
                        )
                    )

    # General grounded fallback
    selected = retrieved_results[:3]

    answer_parts = [
        "Based on the retrieved evidence:"
    ]

    for result in selected:

        text = result.get("text", "").strip()

        if not text:
            continue

        # Keep fallback concise
        if len(text) > 700:
            text = text[:700].rstrip() + "..."

        answer_parts.append(
            f"\n{text}"
        )

    if citations:

        answer_parts.append(
            "\n\nCitations:"
        )

        for citation in citations[:5]:

            answer_parts.append(
                f"- {citation}"
            )

    return "\n".join(answer_parts)


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

    if patient_pdf:

        patient_path = Path(patient_pdf)

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

    print("\n" + "=" * 75)
    print("1. SAFETY CHECK")
    print("=" * 75)

    safety_result = safety_check(
        query=query,
        patient_pdf=(
            patient_path is not None
        )
    )

    if not safety_result["retrieval_allowed"]:

        print("\nSAFETY REFUSAL.")

        print(
            safety_result.get(
                "message",
                "Request refused for safety reasons."
            )
        )

        return build_refusal_result(
            query=query,
            session_id=session_id,
            stage="safety",
            message=safety_result["message"],
            source_mode=None,
            safety=safety_result,
        )

    # ========================================================
    # 2. SCOPE
    # ========================================================

    print("\n" + "=" * 75)
    print("2. SCOPE CHECK")
    print("=" * 75)

    if is_explicitly_out_of_scope(query):

        message = (
            "I couldn't answer this question because "
            "the indexed guideline covers lung cancer "
            "and does not provide evidence about the "
            "condition mentioned in your question."
        )

        print("\nOUT-OF-SCOPE REFUSAL.")
        print(message)

        return build_refusal_result(
            query=query,
            session_id=session_id,
            stage="scope",
            message=message,
            source_mode=SOURCE_CORE,
            safety=safety_result,
        )

    # ========================================================
    # 3. RETRIEVAL
    # ========================================================

    print("\n" + "=" * 75)
    print("3. SOURCE ROUTING + RETRIEVAL")
    print("=" * 75)

    retrieval_data = retrieve_evidence(
        query=query,
        patient_pdf=patient_path,
        session_id=session_id,
    )

    source_mode = retrieval_data["source_mode"]

    core_results = retrieval_data["core_results"]

    patient_results = retrieval_data["patient_results"]

    generation_results = retrieval_data["generation_results"]

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

            metadata = item.get("metadata") or {}

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

            print("Text:")

            print(
                item.get(
                    "text",
                    ""
                )
            )

    # ========================================================
    # 4. NO EVIDENCE
    # ========================================================

    if not generation_results:

        return build_refusal_result(
            query=query,
            session_id=session_id,
            stage="retrieval",
            message=(
                "I couldn't find enough relevant "
                "evidence to answer this question "
                "confidently."
            ),
            source_mode=source_mode,
            safety=safety_result,
            core_results=core_results,
            patient_results=patient_results,
        )

    # ========================================================
    # 5. EVIDENCE CHECK
    # ========================================================

    print("\n" + "=" * 75)
    print("4. EVIDENCE CHECK")
    print("=" * 75)

    evidence_result = check_refusal(
        query=query,
        retrieved_results=generation_results,
    )

    print(
        "Evidence decision:",
        evidence_result.get("decision")
    )

    if (
        evidence_result.get("decision")
        == "insufficient"
    ):

        print("\nEVIDENCE REFUSAL.")

        return build_refusal_result(
            query=query,
            session_id=session_id,
            stage="evidence",
            message=(
                "I couldn't find enough relevant "
                "evidence to answer this question "
                "confidently."
            ),
            source_mode=source_mode,
            safety=safety_result,
            evidence=evidence_result,
            core_results=core_results,
            patient_results=patient_results,
            retrieved_results=generation_results,
        )

    # ========================================================
    # 6. CITATIONS
    # ========================================================

    print("\n" + "=" * 75)
    print("5. CITATIONS")
    print("=" * 75)

    citations = attach_citations(
        generation_results
    )

    # ========================================================
    # 7. GROUNDED PROMPT
    # ========================================================

    print("\n" + "=" * 75)
    print("6. GROUNDED PROMPT")
    print("=" * 75)

    persona = safety_result["persona"]

    if hasattr(persona, "value"):
        persona = persona.value

    patient_priority_instruction = ""

    if source_mode == SOURCE_BOTH:

        patient_priority_instruction = """
============================================================
PATIENT-FIRST RULE
============================================================

The user has an uploaded patient report.

PATIENT evidence is the primary source for
patient-specific facts.

If the patient report contains:
- a measurement
- a test result
- a finding
- an interpretation
- a value

use the patient's actual result.

Do NOT replace a patient value with a NICE
guideline value.

CORE/NICE evidence is secondary context.

Use CORE/NICE to explain what a patient result
means medically when appropriate.

If the patient value is NOT found:
say clearly that the specific value was
not found in the uploaded report.

NEVER invent, estimate, or infer a missing
patient value.
============================================================
"""

    grounded_prompt = build_grounded_prompt(
        query=query,
        retrieved_results=generation_results,
        persona=persona,
        evidence_level=evidence_result[
            "evidence_level"
        ],
    )

    if patient_priority_instruction:

        grounded_prompt = (
            grounded_prompt
            + "\n\n"
            + patient_priority_instruction
        )

    # ========================================================
    # 8. LLM GENERATION
    # ========================================================

    print("\n" + "=" * 75)
    print("7. LLM GENERATION")
    print("=" * 75)

    try:

        generation_result = generate_answer(
            grounded_prompt=grounded_prompt
        )

    except Exception as error:

        error_text = str(error)

        quota_error = (
            "429" in error_text
            or
            "RESOURCE_EXHAUSTED" in error_text
            or
            "quota" in error_text.lower()
        )

        if quota_error:

            print(
                "\nWARNING: LLM quota exhausted."
            )

            print(
                "Using grounded evidence fallback."
            )

            fallback_answer = build_fallback_answer(
                query=query,
                retrieved_results=generation_results,
                citations=citations,
                source_mode=source_mode
            )

            return {

                "status": "success",

                "stage": "generation_fallback",

                "reason": "llm_quota_exhausted",

                "query": query,

                "session_id": session_id,

                "source_mode": source_mode,

                "safety": safety_result,

                "evidence": evidence_result,

                "core_results": core_results,

                "patient_results": patient_results,

                "generation_results":
                    generation_results,

                "retrieved_results":
                    generation_results,

                "citations": citations,

                "grounded_prompt":
                    grounded_prompt,

                "answer":
                    fallback_answer,

                "generation": {

                    "mode": "grounded_fallback",

                    "llm_available": False,

                    "error": error_text,
                },
            }

        raise

    # ========================================================
    # 9. FINAL RESULT
    # ========================================================

    answer = generation_result.get(
        "answer"
    )

    # Prevent empty successful answers
    if not answer or not str(answer).strip():

        print(
            "\nWARNING: LLM returned an empty answer."
        )

        fallback_answer = build_fallback_answer(
            query=query,
            retrieved_results=generation_results,
            citations=citations,
            source_mode=source_mode
        )

        return {

            "status": "success",

            "stage": "generation_fallback",

            "reason": "empty_llm_answer",

            "query": query,

            "session_id": session_id,

            "source_mode": source_mode,

            "safety": safety_result,

            "evidence": evidence_result,

            "core_results": core_results,

            "patient_results": patient_results,

            "generation_results":
                generation_results,

            "retrieved_results":
                generation_results,

            "citations": citations,

            "grounded_prompt":
                grounded_prompt,

            "answer":
                fallback_answer,

            "generation":
                generation_result,
        }

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

        "generation_results":
            generation_results,

        "retrieved_results":
            generation_results,

        "citations": citations,

        "grounded_prompt":
            grounded_prompt,

        "answer":
            answer,

        "generation":
            generation_result,
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_pipeline_result(
    result: Dict[str, Any]
):

    print("\n" + "=" * 75)
    print("PULMO GUIDE - PIPELINE RESULT")
    print("=" * 75)

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

    if result.get("reason"):

        print(
            "Reason:",
            result.get("reason")
        )

    print("\n" + "-" * 75)
    print("CORE RESULTS")
    print("-" * 75)

    core_results = result.get(
        "core_results",
        []
    )

    if not core_results:
        print("No Core results.")

    for item in core_results:

        metadata = item.get("metadata") or {}

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

    print("\n" + "-" * 75)
    print("PATIENT RESULTS")
    print("-" * 75)

    patient_results = result.get(
        "patient_results",
        []
    )

    if not patient_results:
        print("No Patient results.")

    for item in patient_results:

        metadata = item.get("metadata") or {}

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

    print("\n" + "-" * 75)
    print("CITATIONS")
    print("-" * 75)

    citations = result.get(
        "citations",
        []
    )

    if not citations:
        print("No citations.")

    for citation in citations:
        print(citation)

    print("\n" + "-" * 75)
    print("FINAL ANSWER")
    print("-" * 75)

    print(
        result.get(
            "answer",
            ""
        )
    )

    print("\n" + "=" * 75)


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print(
        "PULMO GUIDE - END-TO-END PIPELINE TEST"
    )
    print("=" * 75)

    # ========================================================
    # TEST 1 - CORE
    # ========================================================

    test_query = (
        "What imaging should be offered to people "
        "with stage 3 NSCLC who are having treatment "
        "with curative intent?"
    )

    print("\n" + "=" * 75)
    print("TEST 1 - CORE")
    print("=" * 75)

    try:

        result = run_pipeline(
            query=test_query,
            patient_pdf=None
        )

        print_pipeline_result(result)

    except Exception as error:

        print("\nCORE PIPELINE ERROR:")

        print(
            type(error).__name__,
            ":",
            str(error)
        )

    # ========================================================
    # TEST 2 - PATIENT VALUE
    # ========================================================

    patient_pdf = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "patient"
        / "pulmonary_function_report.pdf"
    )

    session_id = "test_patient_001"

    patient_query_1 = "What is my FEV1?"

    print("\n" + "=" * 75)
    print("TEST 2 - PATIENT VALUE")
    print("=" * 75)

    if patient_pdf.exists():

        try:

            result = run_pipeline(
                query=patient_query_1,
                patient_pdf=str(patient_pdf),
                session_id=session_id
            )

            print_pipeline_result(result)

        except Exception as error:

            print("\nPATIENT PIPELINE ERROR:")

            print(
                type(error).__name__,
                ":",
                str(error)
            )

    else:

        print(
            "\nPatient PDF not found:",
            patient_pdf
        )

    # ========================================================
    # TEST 3 - PATIENT FOLLOW-UP
    # ========================================================

    patient_query_2 = (
        "What does this result mean?"
    )

    print("\n" + "=" * 75)
    print("TEST 3 - PATIENT FOLLOW-UP")
    print("=" * 75)

    if patient_pdf.exists():

        try:

            result = run_pipeline(
                query=patient_query_2,
                patient_pdf=str(patient_pdf),
                session_id=session_id
            )

            print_pipeline_result(result)

        except Exception as error:

            print("\nPATIENT FOLLOW-UP ERROR:")

            print(
                type(error).__name__,
                ":",
                str(error)
            )

    # ========================================================
    # TEST 4 - OUT OF SCOPE
    # ========================================================

    out_of_scope_query = (
        "What is the recommended treatment "
        "for pancreatic cancer according to this guideline?"
    )

    print("\n" + "=" * 75)
    print("TEST 4 - OUT OF SCOPE / REFUSAL")
    print("=" * 75)

    try:

        result = run_pipeline(
            query=out_of_scope_query,
            patient_pdf=None
        )

        print_pipeline_result(result)

    except Exception as error:

        print("\nOUT-OF-SCOPE TEST ERROR:")

        print(
            type(error).__name__,
            ":",
            str(error)
        )

    print("\n" + "=" * 75)
    print("TESTS COMPLETED")
    print("=" * 75)
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
    NEVER stored in Core ChromaDB

PIPELINE:
    1. Safety
    2. Hybrid Scope Gate
    3. Source Routing
    4. Retrieval
    5. Evidence Check
    6. Citations
    7. Grounded Prompt
    8. LLM Generation
    9. Grounded Fallback

============================================================
HYBRID SCOPE DESIGN
============================================================

Scope is determined using THREE layers:

    1. HARD RULES
       - Explicitly unrelated cancers
       - Explicitly unrelated diseases
       - Explicitly unrelated topics
       - These are rejected immediately.

    2. LEXICAL / CONTEXT RULES
       - Explicit lung-cancer terminology
       - Lung + medical context
       - Patient-report terminology
       - Patient-specific medical measurements/findings

    3. SEMANTIC MATCHING
       - BGE sentence embeddings
       - Query is compared with curated
         Pulmo Guide scope prototypes.
       - Semantic similarity helps recognize
         paraphrased questions that do not contain
         exact keyword matches.

IMPORTANT:
    Semantic similarity is NOT allowed to bypass
    hard out-of-scope rules.

    Semantic similarity is also NOT sufficient by itself
    to accept a completely generic question.

Examples:

    "What imaging is recommended for stage III NSCLC?"
        -> CORE

    "How should non-small cell lung cancer be staged?"
        -> CORE

    "What tests are used before surgery for lung cancer?"
        -> CORE

    "What does my FEV1 mean?"
        -> CORE + PATIENT

    "What does this result mean?"
        + uploaded report
        -> CORE + PATIENT

    "What is my favorite color?"
        + uploaded report
        -> OUT OF SCOPE

    "What are the symptoms?"
        -> OUT OF SCOPE

    "What are the symptoms of pancreatic cancer?"
        -> OUT OF SCOPE

============================================================
RETRIEVAL
============================================================

Hybrid retrieval:

    Semantic = 70%
    BM25     = 30%

============================================================
SAFETY
============================================================

- Safety is checked before scope.
- Safety refusal is a valid refusal.
- Scope refusal is a valid refusal.
- Evidence refusal is a valid refusal.
- Out-of-scope questions receive:
      NO retrieval
      NO citations
      NO LLM generation

============================================================
PATIENT SAFETY
============================================================

- Patient evidence has priority for patient-specific facts.
- Patient data NEVER enters Core ChromaDB.
- Patient cache is session/document scoped.
- Missing patient values are never invented.
- NICE values are never substituted for missing
  patient values.

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
import re

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

for path in (
    PROCESSING_DIR,
    RETRIEVAL_DIR,
    GENERATION_DIR,
):
    path_string = str(path)

    if path_string not in sys.path:
        sys.path.append(path_string)


# ============================================================
# EXISTING PROJECT MODULES
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
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

CORE_TOP_K = 5
PATIENT_TOP_K = 5

# Hybrid retrieval:
# 70% Semantic Similarity
# 30% BM25 lexical matching

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CACHE_TTL_SECONDS = 60 * 60 * 4


# ============================================================
# HYBRID SCOPE CONFIGURATION
# ============================================================

# Semantic similarity thresholds.
#
# These are intentionally conservative because this is a
# medical application and semantic similarity must not become
# an uncontrolled bypass around the explicit scope rules.

SCOPE_CORE_SEMANTIC_THRESHOLD = 0.48

SCOPE_PATIENT_SEMANTIC_THRESHOLD = 0.46

# Minimum semantic score required for a lexical/contextual
# signal to be accepted as a hybrid scope match.

SCOPE_HYBRID_SEMANTIC_THRESHOLD = 0.42


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
# SCOPE SEMANTIC PROTOTYPES
# ============================================================
#
# These are NOT answers and are NOT retrieved evidence.
#
# They are only semantic reference descriptions used by the
# scope classifier.
#
# Keeping several paraphrases improves robustness against
# wording variation.
# ============================================================

CORE_SCOPE_PROTOTYPES = [

    # English
    "A question about lung cancer.",
    "A medical question about lung cancer diagnosis.",
    "A medical question about lung cancer symptoms.",
    "A medical question about lung cancer staging.",
    "A medical question about lung cancer treatment.",
    "A medical question about lung cancer surgery.",
    "A medical question about chemotherapy for lung cancer.",
    "A medical question about radiotherapy for lung cancer.",
    "A medical question about immunotherapy for lung cancer.",
    "A medical question about targeted therapy for lung cancer.",
    "A medical question about NSCLC.",
    "A medical question about SCLC.",
    "A medical question about non-small cell lung cancer.",
    "A medical question about small cell lung cancer.",
    "A medical question about lung cancer imaging.",
    "A medical question about lung cancer biopsy.",
    "A medical question about bronchoscopy in lung cancer.",
    "A medical question about PET-CT in lung cancer.",
    "A medical question about brain MRI in lung cancer.",
    "A medical question about molecular testing in lung cancer.",
    "A medical question about biomarkers in lung cancer.",
    "A medical question about EGFR in lung cancer.",
    "A medical question about ALK in lung cancer.",
    "A medical question about ROS1 in lung cancer.",
    "A medical question about PD-L1 in lung cancer.",
    "A question about NICE guidance for lung cancer.",
    "A question about NICE NG122 lung cancer recommendations.",

    # Arabic
    "سؤال طبي عن سرطان الرئة.",
    "سؤال عن أعراض سرطان الرئة.",
    "سؤال عن تشخيص سرطان الرئة.",
    "سؤال عن مراحل سرطان الرئة.",
    "سؤال عن علاج سرطان الرئة.",
    "سؤال عن جراحة سرطان الرئة.",
    "سؤال عن العلاج الكيميائي لسرطان الرئة.",
    "سؤال عن العلاج الإشعاعي لسرطان الرئة.",
    "سؤال عن العلاج المناعي لسرطان الرئة.",
    "سؤال عن سرطان الرئة صغير الخلايا.",
    "سؤال عن سرطان الرئة غير صغير الخلايا.",
    "سؤال عن فحوصات سرطان الرئة.",
    "سؤال عن الأشعة والتصوير في سرطان الرئة.",
    "سؤال عن الخزعة في سرطان الرئة.",
    "سؤال عن الطفرات والمؤشرات الحيوية في سرطان الرئة.",
    "سؤال عن إرشادات NICE لسرطان الرئة.",
]


PATIENT_SCOPE_PROTOTYPES = [

    # English
    "A question about my medical report.",
    "A question about my test result.",
    "A question about my scan result.",
    "A question about my biopsy result.",
    "A question about my pathology report.",
    "A question about my imaging findings.",
    "A question about what my medical result means.",
    "A question about whether my medical finding is normal.",
    "A question about my pulmonary function test.",
    "A question about my FEV1 result.",
    "A question about my FVC result.",
    "A question about my FEV1 FVC ratio.",
    "A question about my TLCO result.",
    "A question about my KCO result.",
    "A question about a patient's tumor size.",
    "A question about a patient's lymph node findings.",
    "A question about a patient's biomarker result.",
    "A question about a patient's mutation result.",
    "A question about a patient's EGFR result.",
    "A question about a patient's ALK result.",
    "A question about a patient's ROS1 result.",
    "A question about a patient's PD-L1 result.",

    # Arabic
    "سؤال عن تقريري الطبي.",
    "سؤال عن نتيجة التحليل الخاص بي.",
    "سؤال عن نتيجة الأشعة الخاصة بي.",
    "سؤال عن نتيجة الخزعة الخاصة بي.",
    "سؤال عن نتيجة الفحص الخاص بي.",
    "سؤال عن ماذا تعني نتيجتي الطبية.",
    "سؤال عن هل النتيجة الخاصة بي طبيعية.",
    "سؤال عن وظائف الرئة الخاصة بي.",
    "سؤال عن نتيجة FEV1 الخاصة بي.",
    "سؤال عن نتيجة FVC الخاصة بي.",
    "سؤال عن نتيجة الفحوصات الخاصة بي.",
    "سؤال عن نتيجة الأشعة أو التصوير الخاصة بي.",
]


# ============================================================
# SCOPE SEMANTIC MODEL CACHE
# ============================================================

_SCOPE_CORE_EMBEDDINGS = None
_SCOPE_PATIENT_EMBEDDINGS = None


def get_scope_embeddings():
    """
    Build semantic prototype embeddings once.

    Returns:
        core_embeddings,
        patient_embeddings
    """

    global _SCOPE_CORE_EMBEDDINGS
    global _SCOPE_PATIENT_EMBEDDINGS

    model = get_embedding_model()

    if _SCOPE_CORE_EMBEDDINGS is None:

        print(
            "\nBuilding semantic CORE scope prototypes..."
        )

        _SCOPE_CORE_EMBEDDINGS = model.encode(
            CORE_SCOPE_PROTOTYPES,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    if _SCOPE_PATIENT_EMBEDDINGS is None:

        print(
            "Building semantic PATIENT scope prototypes..."
        )

        _SCOPE_PATIENT_EMBEDDINGS = model.encode(
            PATIENT_SCOPE_PROTOTYPES,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return (
        _SCOPE_CORE_EMBEDDINGS,
        _SCOPE_PATIENT_EMBEDDINGS,
    )


# ============================================================
# PATIENT QUERY KEYWORDS
# ============================================================
#
# IMPORTANT:
#
# Generic words such as:
#
#     my
#     mine
#
# are NOT enough.
#
# A patient query needs a meaningful report/result/test
# reference or a patient-specific medical measurement/finding.
# ============================================================

PATIENT_KEYWORDS = {

    # --------------------------------------------------------
    # Explicit patient-report references
    # --------------------------------------------------------

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
    "in my test",
    "in my scan",
    "in my biopsy",
    "in my pathology",
    "in my imaging",

    "according to my report",
    "according to my results",
    "according to my test",
    "according to my scan",

    "what does my report",
    "what does my scan",
    "what does my test",
    "what does my biopsy",
    "what does my pathology",
    "what does my result",
    "what do my results",

    # --------------------------------------------------------
    # Contextual references to uploaded report
    # --------------------------------------------------------

    "this result",
    "this finding",
    "this report",
    "these results",
    "these findings",
    "this test",
    "this scan",
    "this biopsy",
    "this pathology",

    "what does this mean",
    "what does this result mean",
    "what does this finding mean",
    "what do these results mean",
    "what do these findings mean",

    "is this normal",
    "is this result normal",
    "is this finding normal",
    "is this test normal",
    "is this scan normal",

    "should i be concerned",
    "is this concerning",

    # --------------------------------------------------------
    # Patient-specific measurements
    # --------------------------------------------------------

    "fev1",
    "fvc",
    "fev1/fvc",
    "fev1 fvc",
    "tlco",
    "kco",
    "pef",

    # --------------------------------------------------------
    # Patient-specific findings
    # --------------------------------------------------------

    "tumor size",
    "tumour size",
    "lesion size",
    "lymph node",
    "lymph nodes",
    "mutation",
    "mutations",
    "biomarker",
    "biomarkers",
    "egfr",
    "alk",
    "ros1",
    "pd-l1",

    # --------------------------------------------------------
    # Arabic
    # --------------------------------------------------------

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
    "حسب نتائجي",

    "ماذا يعني تقريري",
    "ماذا تعني نتيجتي",
    "ماذا يعني التحليل",
    "ماذا تعني النتيجة",

    "النتيجة دي",
    "النتيجة هذه",
    "التحليل ده",
    "التحليل هذا",
    "التقرير ده",
    "التقرير هذا",

    "ده معناه ايه",
    "دي معناها ايه",

    "هل ده طبيعي",
    "هل دي طبيعية",

    "هل النتيجة طبيعية",
    "هل التحليل طبيعي",
    "هل التقرير طبيعي",

    "هل يجب أن أقلق",
    "هل لازم أقلق",
}


# ============================================================
# CORE / LUNG-CANCER DOMAIN KEYWORDS
# ============================================================

CORE_KEYWORDS = {

    # --------------------------------------------------------
    # Lung cancer
    # --------------------------------------------------------

    "lung cancer",
    "lung carcinoma",
    "lung tumour",
    "lung tumor",
    "pulmonary cancer",
    "lung malignancy",

    # --------------------------------------------------------
    # Lung cancer types
    # --------------------------------------------------------

    "nsclc",
    "non-small cell lung cancer",
    "non small cell lung cancer",

    "sclc",
    "small cell lung cancer",

    # --------------------------------------------------------
    # Symptoms
    # --------------------------------------------------------

    "lung cancer symptoms",
    "symptoms of lung cancer",
    "signs of lung cancer",

    "coughing blood",
    "coughing up blood",
    "haemoptysis",
    "hemoptysis",

    "persistent cough lung cancer",
    "persistent cough in lung cancer",

    "unexplained weight loss lung cancer",
    "weight loss in lung cancer",

    # --------------------------------------------------------
    # Diagnosis
    # --------------------------------------------------------

    "lung cancer diagnosis",
    "diagnosis of lung cancer",
    "diagnosing lung cancer",

    "lung biopsy",
    "lung cancer biopsy",
    "biopsy for lung cancer",

    "bronchoscopy for lung cancer",
    "bronchoscopy in lung cancer",

    "ct chest for lung cancer",
    "chest ct for lung cancer",
    "ct scan for lung cancer",

    "pet-ct for lung cancer",
    "pet ct for lung cancer",

    "mri brain for lung cancer",
    "brain mri for lung cancer",

    # --------------------------------------------------------
    # Staging
    # --------------------------------------------------------

    "lung cancer staging",
    "staging lung cancer",

    "stage 1 lung cancer",
    "stage 2 lung cancer",
    "stage 3 lung cancer",
    "stage 4 lung cancer",

    "stage i lung cancer",
    "stage ii lung cancer",
    "stage iii lung cancer",
    "stage iv lung cancer",

    # --------------------------------------------------------
    # Treatment
    # --------------------------------------------------------

    "lung cancer treatment",
    "treatment of lung cancer",
    "treating lung cancer",

    "lung cancer surgery",
    "surgery for lung cancer",

    "lung cancer chemotherapy",
    "chemotherapy for lung cancer",

    "lung cancer radiotherapy",
    "radiotherapy for lung cancer",

    "lung cancer immunotherapy",
    "immunotherapy for lung cancer",

    "targeted therapy lung cancer",
    "targeted treatment for lung cancer",

    "palliative care lung cancer",
    "palliative treatment lung cancer",

    # --------------------------------------------------------
    # NICE
    # --------------------------------------------------------

    "nice ng122",
    "ng122",
    "nice lung cancer",
    "nice guideline lung cancer",
    "lung cancer guideline",

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    "lung cancer recommendation",
    "lung cancer recommendations",

    "recommendation for lung cancer",
    "recommendations for lung cancer",

    # --------------------------------------------------------
    # Molecular / biomarkers
    # --------------------------------------------------------

    "egfr lung cancer",
    "alk lung cancer",
    "ros1 lung cancer",
    "pd-l1 lung cancer",

    "molecular testing lung cancer",
    "molecular testing in lung cancer",

    "biomarker testing lung cancer",

    # --------------------------------------------------------
    # Arabic
    # --------------------------------------------------------

    "سرطان الرئة",
    "سرطان الرئه",
    "أورام الرئة",
    "اورام الرئة",
    "ورم الرئة",
    "ورم في الرئة",

    "اعراض سرطان الرئة",
    "أعراض سرطان الرئة",

    "تشخيص سرطان الرئة",
    "تشخيص سرطان الرئه",

    "علاج سرطان الرئة",
    "علاج سرطان الرئه",

    "مراحل سرطان الرئة",
    "مراحل سرطان الرئه",

    "سرطان الرئة صغير الخلايا",
    "سرطان الرئة غير صغير الخلايا",

    "الفحص لسرطان الرئة",
    "فحص سرطان الرئة",

    "إرشادات سرطان الرئة",
    "ارشادات سرطان الرئة",

    "توصيات سرطان الرئة",
    "توصيات سرطان الرئه",

    "دليل nice",
    "إن جي 122",
    "ان جي 122",
}


# ============================================================
# LUNG CONTEXT TERMS
# ============================================================

LUNG_CONTEXT_TERMS = {

    "lung",
    "pulmonary",
    "nsclc",
    "sclc",
    "non-small cell",
    "small cell lung",
    "lung tumour",
    "lung tumor",
    "lung carcinoma",

    # Arabic
    "الرئة",
    "الرئه",
    "رئوي",
    "رئوية",
    "رئتين",
}


# ============================================================
# CORE MEDICAL CONTEXT TERMS
# ============================================================

CORE_CONTEXT_TERMS = {

    "symptom",
    "symptoms",
    "sign",
    "signs",

    "diagnosis",
    "diagnostic",
    "diagnosing",

    "biopsy",
    "bronchoscopy",

    "staging",
    "stage",

    "treatment",
    "therapy",
    "surgery",
    "chemotherapy",
    "radiotherapy",
    "immunotherapy",

    "screening",
    "imaging",
    "scan",
    "ct",
    "pet",
    "pet-ct",
    "mri",

    "molecular",
    "mutation",
    "biomarker",
    "egfr",
    "alk",
    "ros1",
    "pd-l1",

    "recommendation",
    "recommendations",
    "guideline",

    # Arabic

    "أعراض",
    "اعراض",
    "تشخيص",
    "علاج",
    "جراحة",
    "كيماوي",
    "إشعاع",
    "اشعاع",
    "مناعي",
    "تصوير",
    "أشعة",
    "اشعة",
    "فحص",
    "تحاليل",
    "طفرة",
    "مؤشر حيوي",
    "توصية",
    "توصيات",
}


# ============================================================
# PATIENT MEDICAL CONTEXT TERMS
# ============================================================

PATIENT_MEDICAL_TERMS = {

    # Measurements
    "fev1",
    "fvc",
    "fev1/fvc",
    "fev1 fvc",
    "tlco",
    "kco",
    "pef",

    # Report terminology
    "report",
    "result",
    "results",
    "test",
    "tests",
    "scan",
    "biopsy",
    "pathology",
    "imaging",
    "finding",
    "findings",
    "diagnosis",

    # Cancer/report findings
    "tumor",
    "tumour",
    "lesion",
    "lymph node",
    "lymph nodes",
    "mutation",
    "mutations",
    "biomarker",
    "biomarkers",
    "egfr",
    "alk",
    "ros1",
    "pd-l1",

    # Arabic
    "تقرير",
    "تقارير",
    "نتيجة",
    "نتائج",
    "تحليل",
    "تحاليل",
    "أشعة",
    "اشعة",
    "خزعة",
    "تصوير",
    "فحص",
    "نتيجتي",
    "تقريري",
    "النتيجة",
    "التقرير",
    "التحليل",
    "الخزعة",
    "الأشعة",
}


# ============================================================
# EXPLICIT OUT-OF-SCOPE TERMS
# ============================================================

OUT_OF_SCOPE_TERMS = {

    # --------------------------------------------------------
    # Other cancers
    # --------------------------------------------------------

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
    "pancreatic tumour",
    "breast tumor",
    "breast tumour",
    "colon tumor",
    "colon tumour",
    "prostate tumor",
    "prostate tumour",
    "liver tumor",
    "liver tumour",

    # --------------------------------------------------------
    # Other diseases
    # --------------------------------------------------------

    "rheumatoid arthritis",
    "arthritis",
    "diabetes",
    "hypertension",
    "heart disease",
    "heart failure",
    "kidney disease",
    "liver disease",
    "crohn's disease",
    "ulcerative colitis",
    "multiple sclerosis",
    "parkinson's disease",
    "alzheimer's disease",

    # --------------------------------------------------------
    # Arabic
    # --------------------------------------------------------

    "سرطان البنكرياس",
    "سرطان الثدي",
    "سرطان القولون",
    "سرطان البروستاتا",
    "سرطان الكبد",
    "سرطان المخ",
    "سرطان الجلد",
    "سرطان المعدة",
    "سرطان الغدة الدرقية",

    "التهاب المفاصل الروماتويدي",
    "الروماتويد",
    "السكري",
    "مرض السكري",
    "ارتفاع ضغط الدم",
    "أمراض القلب",
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

    # Arabic

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

    if not query:
        return ""

    query = str(query).lower().strip()

    # Normalize common punctuation
    query = re.sub(
        r"[؟?!.,;:()\[\]{}\"'`]",
        " ",
        query,
    )

    # Normalize repeated whitespace
    query = " ".join(
        query.split()
    )

    return query


# ============================================================
# TERM MATCHING
# ============================================================

def contains_any_term(
    query: str,
    terms,
) -> bool:

    normalized = normalize_query(
        query
    )

    return any(
        term in normalized
        for term in terms
    )


# ============================================================
# MATCHED TERMS
# ============================================================

def get_matched_terms(
    query: str,
    terms,
) -> List[str]:

    normalized = normalize_query(
        query
    )

    matches = []

    for term in terms:

        if term in normalized:

            matches.append(
                term
            )

    return matches


# ============================================================
# EXPLICIT OUT-OF-SCOPE CHECK
# ============================================================

def is_explicitly_out_of_scope(
    query: str,
) -> bool:

    normalized = normalize_query(
        query
    )

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
# PATIENT QUERY CHECK — LEXICAL
# ============================================================

def is_patient_query_lexical(
    query: str,
) -> bool:
    """
    Conservative lexical patient detector.

    A generic "my" or "mine" is NOT enough.

    Patient classification requires either:

        1. Explicit patient-report phrase.

        OR

        2. Patient-style contextual phrase.

        OR

        3. Medical patient term.

    This prevents:

        "What is my favorite color?"

    from becoming a patient query merely because
    the word "my" appears.
    """

    normalized = normalize_query(
        query
    )

    # --------------------------------------------------------
    # Explicit report/result references
    # --------------------------------------------------------

    if any(
        keyword in normalized
        for keyword in PATIENT_KEYWORDS
    ):
        return True

    # --------------------------------------------------------
    # Medical measurement / finding references
    # --------------------------------------------------------

    if any(
        term in normalized
        for term in PATIENT_MEDICAL_TERMS
    ):

        # If the query contains a clear first-person/
        # contextual patient reference, classify as patient.

        patient_reference_patterns = [

            "my ",
            "mine ",
            "me ",
            "i have",
            "i got",
            "i received",
            "my value",
            "my measurement",
            "my level",

            # Arabic
            "بتاعي",
            "بتاعتي",
            "بتوعي",
            "عندي",
            "بتاعى",
            "نتيجتي",
            "تقريري",
            "تحليلي",
            "أشعتي",
        ]

        if any(
            pattern in normalized
            for pattern in patient_reference_patterns
        ):
            return True

    return False


# ============================================================
# PATIENT QUERY CHECK — SEMANTIC
# ============================================================

def get_patient_semantic_score(
    query: str,
) -> float:

    core_embeddings, patient_embeddings = (
        get_scope_embeddings()
    )

    del core_embeddings

    model = get_embedding_model()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    scores = np.dot(
        patient_embeddings,
        query_embedding,
    )

    if scores.size == 0:
        return 0.0

    return float(
        np.max(scores)
    )


# ============================================================
# PATIENT QUERY CHECK — HYBRID
# ============================================================

def is_patient_query(
    query: str,
) -> bool:
    """
    Hybrid patient-query classifier.

    Patient lexical signal + semantic similarity.

    Semantic similarity alone is intentionally insufficient.

    This protects against:

        "What is my favorite color?"

    while allowing paraphrases such as:

        "Can you explain the lung function number
         in the report I uploaded?"
    """

    normalized = normalize_query(
        query
    )

    # Strong explicit patient lexical signal.
    lexical_patient = is_patient_query_lexical(
        normalized
    )

    # If strong explicit patient language exists,
    # accept immediately.

    if lexical_patient:
        return True

    # --------------------------------------------------------
    # Semantic patient detection
    # --------------------------------------------------------

    has_medical_term = contains_any_term(
        normalized,
        PATIENT_MEDICAL_TERMS,
    )

    if not has_medical_term:
        return False

    semantic_score = get_patient_semantic_score(
        normalized
    )

    return (
        semantic_score
        >= SCOPE_PATIENT_SEMANTIC_THRESHOLD
    )


# ============================================================
# CORE QUERY CHECK — LEXICAL
# ============================================================

def is_core_query_lexical(
    query: str,
) -> bool:

    normalized = normalize_query(
        query
    )

    # --------------------------------------------------------
    # Strong explicit lung-cancer terms
    # --------------------------------------------------------

    if any(
        keyword in normalized
        for keyword in CORE_KEYWORDS
    ):
        return True

    # --------------------------------------------------------
    # Contextual detection
    #
    # Example:
    #
    # "How is lung cancer diagnosed?"
    #
    # lung context + medical context
    # --------------------------------------------------------

    has_lung_context = any(
        term in normalized
        for term in LUNG_CONTEXT_TERMS
    )

    has_core_context = any(
        term in normalized
        for term in CORE_CONTEXT_TERMS
    )

    if (
        has_lung_context
        and has_core_context
    ):
        return True

    return False


# ============================================================
# CORE SEMANTIC SCORE
# ============================================================

def get_core_semantic_score(
    query: str,
) -> float:

    core_embeddings, patient_embeddings = (
        get_scope_embeddings()
    )

    del patient_embeddings

    model = get_embedding_model()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    scores = np.dot(
        core_embeddings,
        query_embedding,
    )

    if scores.size == 0:
        return 0.0

    return float(
        np.max(scores)
    )


# ============================================================
# CORE QUERY CHECK — HYBRID
# ============================================================

def is_core_query(
    query: str,
) -> bool:
    """
    Hybrid lung-cancer scope classifier.

    Decision strategy:

        1. Exact/lexical core match
           -> ACCEPT

        2. Lung context + medical context
           -> ACCEPT

        3. Semantic similarity + medical/lung signal
           -> ACCEPT

        4. Semantic similarity without meaningful domain
           -> REJECT

    This is intentionally conservative.
    """

    normalized = normalize_query(
        query
    )

    # --------------------------------------------------------
    # Strong lexical signal
    # --------------------------------------------------------

    if is_core_query_lexical(
        normalized
    ):
        return True

    # --------------------------------------------------------
    # Semantic score
    # --------------------------------------------------------

    semantic_score = get_core_semantic_score(
        normalized
    )

    # --------------------------------------------------------
    # Context signals
    # --------------------------------------------------------

    has_lung_context = contains_any_term(
        normalized,
        LUNG_CONTEXT_TERMS,
    )

    has_core_context = contains_any_term(
        normalized,
        CORE_CONTEXT_TERMS,
    )

    # --------------------------------------------------------
    # Hybrid decision
    #
    # Semantic similarity is allowed to recover
    # paraphrased domain questions, but only if the
    # query has a meaningful medical/domain signal.
    # --------------------------------------------------------

    if (
        has_lung_context
        and semantic_score
        >= SCOPE_HYBRID_SEMANTIC_THRESHOLD
    ):
        return True

    if (
        has_core_context
        and semantic_score
        >= SCOPE_CORE_SEMANTIC_THRESHOLD
    ):
        return True

    return False


# ============================================================
# HYBRID SCOPE CLASSIFICATION DETAILS
# ============================================================

def classify_scope(
    query: str,
    patient_pdf: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Return detailed hybrid scope information.

    This is useful for debugging and evaluation.

    Returned fields include:

        in_scope
        explicit_out_of_scope
        patient_query
        core_query
        core_semantic_score
        patient_semantic_score
        matched_core_terms
        matched_patient_terms
        decision
        reason
    """

    normalized = normalize_query(
        query
    )

    if not normalized:

        return {

            "in_scope": False,
            "explicit_out_of_scope": False,
            "patient_query": False,
            "core_query": False,
            "core_semantic_score": 0.0,
            "patient_semantic_score": 0.0,
            "matched_core_terms": [],
            "matched_patient_terms": [],
            "decision": "out_of_scope",
            "reason": "empty_query",
        }

    # --------------------------------------------------------
    # Hard rejection
    # --------------------------------------------------------

    explicit_oos = is_explicitly_out_of_scope(
        normalized
    )

    if explicit_oos:

        return {

            "in_scope": False,
            "explicit_out_of_scope": True,
            "patient_query": False,
            "core_query": False,
            "core_semantic_score": 0.0,
            "patient_semantic_score": 0.0,
            "matched_core_terms": [],
            "matched_patient_terms": [],
            "decision": "out_of_scope",
            "reason": "explicit_out_of_scope_term",
        }

    # --------------------------------------------------------
    # Lexical matches
    # --------------------------------------------------------

    matched_core_terms = get_matched_terms(
        normalized,
        CORE_KEYWORDS,
    )

    matched_patient_terms = get_matched_terms(
        normalized,
        PATIENT_KEYWORDS,
    )

    # --------------------------------------------------------
    # Determine whether patient semantic analysis
    # is needed.
    #
    # We only perform patient semantic matching when
    # a patient report exists.
    # --------------------------------------------------------

    patient_query = False
    patient_semantic_score = 0.0

    if patient_pdf is not None:

        patient_query = is_patient_query(
            normalized
        )

        if (
            not patient_query
            and contains_any_term(
                normalized,
                PATIENT_MEDICAL_TERMS,
            )
        ):

            patient_semantic_score = (
                get_patient_semantic_score(
                    normalized
                )
            )

            if (
                patient_semantic_score
                >= SCOPE_PATIENT_SEMANTIC_THRESHOLD
            ):

                patient_query = True

    # --------------------------------------------------------
    # Core classification
    # --------------------------------------------------------

    core_query = is_core_query(
        normalized
    )

    core_semantic_score = 0.0

    if not core_query:

        # Only calculate semantic score if lexical
        # classification did not already succeed.

        core_semantic_score = (
            get_core_semantic_score(
                normalized
            )
        )

    else:

        # For diagnostics, still calculate score only
        # when useful.
        #
        # This avoids unnecessary model work in the
        # strongest lexical cases.

        core_semantic_score = 0.0

    # --------------------------------------------------------
    # Patient-specific query
    # --------------------------------------------------------

    if patient_pdf is not None and patient_query:

        return {

            "in_scope": True,

            "explicit_out_of_scope": False,

            "patient_query": True,

            "core_query": core_query,

            "core_semantic_score":
                core_semantic_score,

            "patient_semantic_score":
                patient_semantic_score,

            "matched_core_terms":
                matched_core_terms,

            "matched_patient_terms":
                matched_patient_terms,

            "decision":
                "patient",

            "reason":
                "patient_report_query",
        }

    # --------------------------------------------------------
    # Core query
    # --------------------------------------------------------

    if core_query:

        return {

            "in_scope": True,

            "explicit_out_of_scope": False,

            "patient_query": False,

            "core_query": True,

            "core_semantic_score":
                core_semantic_score,

            "patient_semantic_score":
                patient_semantic_score,

            "matched_core_terms":
                matched_core_terms,

            "matched_patient_terms":
                matched_patient_terms,

            "decision":
                "core",

            "reason":
                "lung_cancer_domain_query",
        }

    # --------------------------------------------------------
    # Everything else
    # --------------------------------------------------------

    return {

        "in_scope": False,

        "explicit_out_of_scope": False,

        "patient_query": False,

        "core_query": False,

        "core_semantic_score":
            core_semantic_score,

        "patient_semantic_score":
            patient_semantic_score,

        "matched_core_terms":
            matched_core_terms,

        "matched_patient_terms":
            matched_patient_terms,

        "decision":
            "out_of_scope",

        "reason":
            "no_sufficient_scope_signal",
    }


# ============================================================
# FINAL SCOPE CHECK
# ============================================================

def is_query_in_scope(
    query: str,
    patient_pdf: Optional[Path] = None,
) -> bool:
    """
    Hybrid scope gate.

    IMPORTANT:
        This function executes BEFORE retrieval.
    """

    classification = classify_scope(
        query=query,
        patient_pdf=patient_pdf,
    )

    return bool(
        classification.get(
            "in_scope",
            False,
        )
    )


# ============================================================
# OUT-OF-SCOPE MESSAGE
# ============================================================

def build_scope_refusal_message(
    query: str,
) -> str:

    return (
        "This question is outside the scope of the indexed "
        "guidelines and uploaded patient report."
    )


# ============================================================
# FILE HASH
# ============================================================

def file_hash(
    file_path: Path,
) -> str:

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb",
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
    session_id: str,
) -> Path:

    document_hash = file_hash(
        patient_pdf
    )

    cache_id = (
        f"{session_id}_"
        f"{document_hash[:16]}"
    )

    cache_dir = (
        PATIENT_CACHE_DIR /
        cache_id
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return cache_dir


# ============================================================
# CACHE VALIDATION
# ============================================================

def cache_is_valid(
    cache_dir: Path,
    patient_pdf: Optional[Path] = None,
    session_id: Optional[str] = None,
) -> bool:

    metadata_path = (
        cache_dir /
        "cache_metadata.json"
    )

    chunks_path = (
        cache_dir /
        "patient_chunks.json"
    )

    if not metadata_path.exists():
        return False

    if not chunks_path.exists():
        return False

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(
                file
            )

    except Exception:

        return False

    created_at = metadata.get(
        "created_at",
        0,
    )

    if not created_at:
        return False

    age = time.time() - created_at

    if age > CACHE_TTL_SECONDS:
        return False

    if session_id is not None:

        if metadata.get(
            "session_id"
        ) != session_id:

            return False

    if patient_pdf is not None:

        try:

            current_hash = file_hash(
                patient_pdf
            )

        except Exception:

            return False

        cached_hash = metadata.get(
            "document_hash"
        )

        if (
            not cached_hash
            or cached_hash != current_hash
        ):

            return False

    return True


# ============================================================
# SAVE CACHE METADATA
# ============================================================

def save_cache_metadata(
    cache_dir: Path,
    patient_pdf: Path,
    session_id: str,
) -> None:

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

        "semantic_weight":
            SEMANTIC_WEIGHT,

        "bm25_weight":
            BM25_WEIGHT,

        "cache_ttl_seconds":
            CACHE_TTL_SECONDS,
    }

    metadata_path = (
        cache_dir /
        "cache_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# SOURCE ROUTER
# ============================================================

def determine_source_mode(
    query: str,
    patient_pdf: Optional[Path],
) -> str:
    """
    Determine which knowledge source(s) should be searched.

    Rules:

        Core question + no patient PDF
            -> CORE

        Core question + patient PDF
            -> CORE

        Patient-specific question + patient PDF
            -> CORE + PATIENT

        Patient-specific question without patient PDF
            -> CORE

            NOTE:
            The query should normally be rejected by the
            scope gate before reaching this case.
    """

    patient_query = is_patient_query(
        query
    )

    core_query = is_core_query(
        query
    )

    # --------------------------------------------------------
    # Patient-specific question
    # --------------------------------------------------------

    if patient_query:

        if patient_pdf is not None:
            return SOURCE_BOTH

        return SOURCE_CORE

    # --------------------------------------------------------
    # Core question
    # --------------------------------------------------------

    if core_query:
        return SOURCE_CORE

    # --------------------------------------------------------
    # Defensive fallback
    # --------------------------------------------------------

    if patient_pdf is not None:
        return SOURCE_PATIENT

    return SOURCE_CORE


# ============================================================
# CORE RETRIEVAL
# ============================================================

def retrieve_core(
    query: str,
) -> List[Dict[str, Any]]:

    results = hybrid_search(
        query=query,
        final_top_k=CORE_TOP_K,
    )

    normalized_results = []

    for result in results:

        result = dict(
            result
        )

        metadata = dict(
            result.get(
                "metadata"
            ) or {}
        )

        metadata[
            "source_type"
        ] = SOURCE_CORE

        result[
            "metadata"
        ] = metadata

        result[
            "source_type"
        ] = SOURCE_CORE

        normalized_results.append(
            result
        )

    return normalized_results


# ============================================================
# PATIENT PROCESSING
# ============================================================

def process_patient_report(
    patient_pdf: Path,
    session_id: str,
) -> Dict[str, Any]:

    patient_pdf = Path(
        patient_pdf
    )

    if not patient_pdf.exists():

        raise FileNotFoundError(
            "Patient report not found: "
            f"{patient_pdf}"
        )

    cache_dir = get_patient_cache_dir(
        patient_pdf,
        session_id,
    )

    chunks_path = (
        cache_dir /
        "patient_chunks.json"
    )

    # ========================================================
    # CACHE HIT
    # ========================================================

    if cache_is_valid(
        cache_dir,
        patient_pdf=patient_pdf,
        session_id=session_id,
    ):

        print(
            "\nPatient cache HIT."
        )

        with open(
            chunks_path,
            "r",
            encoding="utf-8",
        ) as file:

            chunks = json.load(
                file
            )

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
            "Patient PDF produced no parsed elements."
        )

    print(
        f"   Parsed elements: "
        f"{len(elements)}"
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
            "Patient cleaning produced no elements."
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
            "Patient section detection produced "
            "no elements."
        )

    print(
        f"   Sectioned elements: "
        f"{len(sectioned_elements)}"
    )

    # ========================================================
    # 4. EMBEDDING MODEL
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
        session_id=session_id,
    )

    if not chunks:

        raise ValueError(
            "Patient chunking produced no chunks."
        )

    print(
        f"   Created chunks: "
        f"{len(chunks)}"
    )

    # ========================================================
    # 6. NORMALIZE PATIENT METADATA
    # ========================================================

    normalized_chunks = []

    for chunk in chunks:

        chunk = dict(
            chunk
        )

        metadata = dict(
            chunk.get(
                "metadata"
            ) or {}
        )

        metadata[
            "source_type"
        ] = SOURCE_PATIENT

        metadata[
            "session_id"
        ] = session_id

        metadata.setdefault(
            "document_name",
            patient_pdf.name,
        )

        chunk[
            "metadata"
        ] = metadata

        chunk[
            "source_type"
        ] = SOURCE_PATIENT

        normalized_chunks.append(
            chunk
        )

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
        encoding="utf-8",
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    save_cache_metadata(
        cache_dir,
        patient_pdf,
        session_id,
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
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(
    scores,
) -> np.ndarray:

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if scores.size == 0:
        return scores

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:

        return np.ones_like(
            scores
        )

    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# PATIENT HYBRID RETRIEVAL
# ============================================================

def patient_hybrid_search(
    query: str,
    chunks: List[Dict[str, Any]],
    final_top_k: int = PATIENT_TOP_K,
) -> List[Dict[str, Any]]:

    if not chunks:
        return []

    texts = [
        str(
            chunk.get(
                "text",
                ""
            )
        )
        for chunk in chunks
    ]

    valid_indices = [
        index
        for index, text in enumerate(texts)
        if text.strip()
    ]

    if not valid_indices:
        return []

    valid_texts = [
        texts[index]
        for index in valid_indices
    ]

    valid_chunks = [
        chunks[index]
        for index in valid_indices
    ]

    # ========================================================
    # SEMANTIC RETRIEVAL
    # ========================================================

    model = get_embedding_model()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    document_embeddings = model.encode(
        valid_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    semantic_scores = np.dot(
        document_embeddings,
        query_embedding,
    )

    # ========================================================
    # BM25 RETRIEVAL
    # ========================================================

    tokenized_documents = [
        text.lower().split()
        for text in valid_texts
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
    # SCORE NORMALIZATION
    # ========================================================

    semantic_normalized = normalize_scores(
        semantic_scores
    )

    bm25_normalized = normalize_scores(
        bm25_scores
    )

    # ========================================================
    # HYBRID SCORE
    # ========================================================

    hybrid_scores = (
        SEMANTIC_WEIGHT
        * semantic_normalized
        +
        BM25_WEIGHT
        * bm25_normalized
    )

    # ========================================================
    # TOP K
    # ========================================================

    top_k = min(
        final_top_k,
        len(valid_chunks),
    )

    indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        indices,
        start=1,
    ):

        chunk = dict(
            valid_chunks[index]
        )

        metadata = dict(
            chunk.get(
                "metadata"
            ) or {}
        )

        metadata[
            "source_type"
        ] = SOURCE_PATIENT

        chunk[
            "metadata"
        ] = metadata

        chunk[
            "source_type"
        ] = SOURCE_PATIENT

        results.append({

            "hybrid_rank":
                rank,

            "chunk_id":
                chunk.get(
                    "chunk_id",
                    f"patient_{index}",
                ),

            "text":
                chunk.get(
                    "text",
                    "",
                ),

            "metadata":
                metadata,

            "source_type":
                SOURCE_PATIENT,

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
                ),
        })

    return results


# ============================================================
# PATIENT RETRIEVAL
# ============================================================

def retrieve_patient(
    query: str,
    patient_pdf: Path,
    session_id: str,
) -> List[Dict[str, Any]]:

    patient_data = process_patient_report(
        patient_pdf,
        session_id,
    )

    chunks = patient_data.get(
        "chunks",
        [],
    )

    if not chunks:
        return []

    results = patient_hybrid_search(
        query=query,
        chunks=chunks,
        final_top_k=PATIENT_TOP_K,
    )

    normalized_results = []

    for result in results:

        result = dict(
            result
        )

        metadata = dict(
            result.get(
                "metadata"
            ) or {}
        )

        metadata[
            "source_type"
        ] = SOURCE_PATIENT

        metadata[
            "session_id"
        ] = session_id

        metadata.setdefault(
            "document_name",
            patient_pdf.name,
        )

        result[
            "metadata"
        ] = metadata

        result[
            "source_type"
        ] = SOURCE_PATIENT

        normalized_results.append(
            result
        )

    return normalized_results


# ============================================================
# GENERATION ORDER
# ============================================================

def build_generation_results(
    source_mode: str,
    core_results: List[Dict[str, Any]],
    patient_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    # Patient evidence MUST come first for patient-specific
    # questions.

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
    session_id: Optional[str] = None,
) -> Dict[str, Any]:

    source_mode = determine_source_mode(
        query,
        patient_pdf,
    )

    core_results = []
    patient_results = []

    # ========================================================
    # CORE
    # ========================================================

    if source_mode in (
        SOURCE_CORE,
        SOURCE_BOTH,
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
        source_mode in (
            SOURCE_PATIENT,
            SOURCE_BOTH,
        )
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
            session_id=session_id,
        )

    generation_results = build_generation_results(
        source_mode=source_mode,
        core_results=core_results,
        patient_results=patient_results,
    )

    return {

        "source_mode":
            source_mode,

        "core_results":
            core_results,

        "patient_results":
            patient_results,

        "combined_results":
            core_results + patient_results,

        "generation_results":
            generation_results,
    }


# ============================================================
# CITATIONS
# ============================================================

def attach_citations(
    retrieved_results: List[Dict[str, Any]],
) -> List[str]:

    citations = []

    for result in retrieved_results:

        metadata = dict(
            result.get(
                "metadata"
            ) or {}
        )

        citation = build_citation(
            metadata
        )

        result[
            "citation"
        ] = citation

        if (
            citation
            and citation not in citations
        ):

            citations.append(
                citation
            )

    return citations


# ============================================================
# CLEAN PATIENT SESSION
# ============================================================

def cleanup_patient_session(
    session_id: str,
) -> None:

    if not session_id:
        return

    for cache_dir in PATIENT_CACHE_DIR.glob(
        f"{session_id}_*"
    ):

        if cache_dir.exists():

            shutil.rmtree(
                cache_dir,
                ignore_errors=True,
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

        "status":
            "refused",

        "stage":
            stage,

        "query":
            query,

        "session_id":
            session_id,

        "source_mode":
            source_mode,

        "safety":
            safety,

        "evidence":
            evidence,

        "core_results":
            core_results or [],

        "patient_results":
            patient_results or [],

        "retrieved_results":
            retrieved_results or [],

        "citations":
            [],

        "grounded_prompt":
            None,

        "answer":
            message,

        "generation":
            None,
    }


# ============================================================
# GROUNDED FALLBACK
# ============================================================

def build_fallback_answer(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    citations: List[str],
    source_mode: str,
) -> str:

    if not retrieved_results:

        return (
            "I couldn't find enough relevant evidence "
            "to answer this question confidently."
        )

    patient_results = [
        result
        for result in retrieved_results
        if result.get(
            "source_type"
        ) == SOURCE_PATIENT
    ]

    # ========================================================
    # PATIENT-SPECIFIC VALUE FALLBACK
    # ========================================================

    normalized = normalize_query(
        query
    )

    patient_value_terms = {
        "fev1",
        "fvc",
        "fev1/fvc",
        "tlco",
        "kco",
        "pef",
    }

    requested_patient_value = next(
        (
            term
            for term in patient_value_terms
            if term in normalized
        ),
        None,
    )

    if (
        source_mode == SOURCE_BOTH
        and patient_results
        and requested_patient_value
    ):

        patient_text = "\n".join(
            result.get(
                "text",
                ""
            )
            for result in patient_results[:3]
        )

        patterns = {

            "fev1": [

                r"FEV1\s*(?:\(L\))?"
                r"\s*[:\-]?\s*"
                r"([0-9]+(?:\.[0-9]+)?)\s*L",

                r"FEV1"
                r".{0,100}?"
                r"([0-9]{2,3})\s*%"
                r"\s*(?:predicted)?",
            ],

            "fvc": [

                r"FVC\s*(?:\(L\))?"
                r"\s*[:\-]?\s*"
                r"([0-9]+(?:\.[0-9]+)?)\s*L",

                r"FVC"
                r".{0,100}?"
                r"([0-9]{2,3})\s*%"
                r"\s*(?:predicted)?",
            ],

            "fev1/fvc": [

                r"FEV1\s*/\s*FVC"
                r".{0,100}?"
                r"([0-9]+(?:\.[0-9]+)?)\s*%?",
            ],

            "tlco": [

                r"TLCO"
                r".{0,100}?"
                r"([0-9]+(?:\.[0-9]+)?)\s*%"
                r"?",
            ],

            "kco": [

                r"KCO"
                r".{0,100}?"
                r"([0-9]+(?:\.[0-9]+)?)\s*%"
                r"?",
            ],

            "pef": [

                r"PEF"
                r".{0,100}?"
                r"([0-9]+(?:\.[0-9]+)?)",
            ],
        }

        for pattern in patterns.get(
            requested_patient_value,
            [],
        ):

            match = re.search(
                pattern,
                patient_text,
                re.IGNORECASE | re.DOTALL,
            )

            if match:

                value = match.group(
                    1
                )

                patient_citations = []

                for result in patient_results:

                    citation = result.get(
                        "citation"
                    )

                    if (
                        citation
                        and citation
                        not in patient_citations
                    ):

                        patient_citations.append(
                            citation
                        )

                answer = (
                    "According to the uploaded "
                    "patient report, the retrieved "
                    f"{requested_patient_value.upper()} "
                    f"value is {value}."
                )

                if patient_citations:

                    answer += (
                        "\n\nCitations:\n"
                        +
                        "\n".join(
                            f"- {citation}"
                            for citation
                            in patient_citations[:3]
                        )
                    )

                return answer

    # ========================================================
    # GENERAL GROUNDED FALLBACK
    # ========================================================

    selected_results = (
        retrieved_results[:3]
    )

    answer_parts = [
        "Based on the retrieved evidence:"
    ]

    for result in selected_results:

        text = str(
            result.get(
                "text",
                ""
            )
        ).strip()

        if not text:
            continue

        if len(text) > 700:

            text = (
                text[:700]
                .rstrip()
                + "..."
            )

        source_type = result.get(
            "source_type"
        )

        if source_type == SOURCE_PATIENT:

            prefix = (
                "Patient report evidence: "
            )

        else:

            prefix = (
                "NICE/Core evidence: "
            )

        answer_parts.append(
            f"\n{prefix}{text}"
        )

    if citations:

        answer_parts.append(
            "\n\nCitations:"
        )

        for citation in citations[:5]:

            answer_parts.append(
                f"- {citation}"
            )

    return "\n".join(
        answer_parts
    )


# ============================================================
# QUOTA ERROR DETECTION
# ============================================================

def is_llm_quota_error(
    error: Exception,
) -> bool:

    error_text = str(
        error
    )

    return (
        "429" in error_text
        or
        "RESOURCE_EXHAUSTED"
        in error_text
        or
        "quota"
        in error_text.lower()
        or
        "rate limit"
        in error_text.lower()
    )


# ============================================================
# BUILD SUCCESS RESULT
# ============================================================

def build_success_result(
    *,
    stage: str,
    query: str,
    session_id: Optional[str],
    source_mode: str,
    safety_result: Dict[str, Any],
    evidence_result: Dict[str, Any],
    core_results: List[Dict[str, Any]],
    patient_results: List[Dict[str, Any]],
    generation_results: List[Dict[str, Any]],
    citations: List[str],
    grounded_prompt: str,
    answer: str,
    generation: Any,
    reason: Optional[str] = None,
) -> Dict[str, Any]:

    result = {

        "status":
            "success",

        "stage":
            stage,

        "query":
            query,

        "session_id":
            session_id,

        "source_mode":
            source_mode,

        "safety":
            safety_result,

        "evidence":
            evidence_result,

        "core_results":
            core_results,

        "patient_results":
            patient_results,

        "generation_results":
            generation_results,

        "retrieved_results":
            generation_results,

        "citations":
            citations,

        "grounded_prompt":
            grounded_prompt,

        "answer":
            answer,

        "generation":
            generation,
    }

    if reason:

        result["reason"] = reason

    return result


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    query: str,
    patient_pdf: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main Pulmo Guide end-to-end pipeline.

    Order:

        1. Safety
        2. Hybrid Scope Gate
        3. Source Routing
        4. Retrieval
        5. Evidence Check
        6. Citation Attachment
        7. Grounded Prompt
        8. LLM Generation
        9. Grounded Fallback
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    patient_path = None

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
        ),
    )

    if not safety_result.get(
        "retrieval_allowed",
        False,
    ):

        message = safety_result.get(
            "message",
            "Request refused for safety reasons.",
        )

        print(
            "\nSAFETY REFUSAL."
        )

        print(
            message
        )

        return build_refusal_result(
            query=query,
            session_id=session_id,
            stage="safety",
            message=message,
            source_mode=None,
            safety=safety_result,
        )

    # ========================================================
    # 2. HYBRID SCOPE GATE
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "2. HYBRID SCOPE CHECK"
    )

    print(
        "=" * 75
    )

    scope_result = classify_scope(
        query=query,
        patient_pdf=patient_path,
    )

    in_scope = scope_result.get(
        "in_scope",
        False,
    )

    print(
        "\nScope decision:",
        scope_result.get(
            "decision"
        ),
    )

    print(
        "Scope reason:",
        scope_result.get(
            "reason"
        ),
    )

    print(
        "Core semantic score:",
        round(
            scope_result.get(
                "core_semantic_score",
                0.0,
            ),
            4,
        ),
    )

    print(
        "Patient semantic score:",
        round(
            scope_result.get(
                "patient_semantic_score",
                0.0,
            ),
            4,
        ),
    )

    print(
        "Matched core terms:",
        scope_result.get(
            "matched_core_terms",
            [],
        ),
    )

    print(
        "Matched patient terms:",
        scope_result.get(
            "matched_patient_terms",
            [],
        ),
    )

    print(
        "In scope:",
        in_scope,
    )

    # --------------------------------------------------------
    # CRITICAL:
    #
    # OUT OF SCOPE = STOP
    #
    # No retrieval.
    # No citations.
    # No prompt.
    # No LLM.
    # --------------------------------------------------------

    if not in_scope:

        message = build_scope_refusal_message(
            query
        )

        print(
            "\nOUT-OF-SCOPE REFUSAL."
        )

        print(
            message
        )

        return build_refusal_result(
            query=query,
            session_id=session_id,
            stage="scope",
            message=message,
            source_mode=None,
            safety=safety_result,
            core_results=[],
            patient_results=[],
            retrieved_results=[],
        )

    # ========================================================
    # 3. SOURCE ROUTING
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "3. SOURCE ROUTING"
    )

    print(
        "=" * 75
    )

    source_mode = determine_source_mode(
        query=query,
        patient_pdf=patient_path,
    )

    print(
        f"Selected source mode: {source_mode}"
    )

    # ========================================================
    # 4. RETRIEVAL
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "4. RETRIEVAL"
    )

    print(
        "=" * 75
    )

    retrieval_data = retrieve_evidence(
        query=query,
        patient_pdf=patient_path,
        session_id=session_id,
    )

    core_results = retrieval_data[
        "core_results"
    ]

    patient_results = retrieval_data[
        "patient_results"
    ]

    generation_results = retrieval_data[
        "generation_results"
    ]

    source_mode = retrieval_data[
        "source_mode"
    ]

    print(
        f"\nSource mode: {source_mode}"
    )

    print(
        f"Core results: "
        f"{len(core_results)}"
    )

    print(
        f"Patient results: "
        f"{len(patient_results)}"
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
            start=1,
        ):

            metadata = item.get(
                "metadata"
            ) or {}

            print(
                f"\n[{index}] "
                f"{item.get('chunk_id')}"
            )

            print(
                "Section:",
                metadata.get(
                    "section",
                    "",
                ),
            )

            print(
                "Page:",
                metadata.get(
                    "page_start",
                    metadata.get(
                        "page",
                        "",
                    ),
                ),
            )

            print(
                "Score:",
                item.get(
                    "hybrid_score",
                    "",
                ),
            )

            print(
                "Text:"
            )

            print(
                item.get(
                    "text",
                    "",
                )
            )

    # ========================================================
    # 5. NO EVIDENCE
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
    # 6. EVIDENCE CHECK
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "5. EVIDENCE CHECK"
    )

    print(
        "=" * 75
    )

    evidence_result = check_refusal(
        query=query,
        retrieved_results=generation_results,
    )

    print(
        "Evidence decision:",
        evidence_result.get(
            "decision"
        ),
    )

    if (
        evidence_result.get(
            "decision"
        )
        == "insufficient"
    ):

        message = (
            "I couldn't find enough relevant "
            "evidence to answer this question "
            "confidently."
        )

        print(
            "\nEVIDENCE REFUSAL."
        )

        return build_refusal_result(
            query=query,
            session_id=session_id,
            stage="evidence",
            message=message,
            source_mode=source_mode,
            safety=safety_result,
            evidence=evidence_result,
            core_results=core_results,
            patient_results=patient_results,
            retrieved_results=generation_results,
        )

    # ========================================================
    # 7. CITATIONS
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "6. CITATIONS"
    )

    print(
        "=" * 75
    )

    citations = attach_citations(
        generation_results
    )

    print(
        f"Generated citations: "
        f"{len(citations)}"
    )

    # ========================================================
    # 8. GROUNDED PROMPT
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "7. GROUNDED PROMPT"
    )

    print(
        "=" * 75
    )

    persona = safety_result.get(
        "persona",
        "general",
    )

    if hasattr(
        persona,
        "value",
    ):

        persona = persona.value

    grounded_prompt = build_grounded_prompt(
        query=query,
        retrieved_results=generation_results,
        persona=persona,
        evidence_level=evidence_result.get(
            "evidence_level"
        ),
    )

    # --------------------------------------------------------
    # Patient-first instruction
    # --------------------------------------------------------

    if source_mode == SOURCE_BOTH:

        patient_priority_instruction = """
============================================================
PATIENT-FIRST GROUNDING RULE
============================================================

The user has uploaded a patient report.

For patient-specific facts:

1. PATIENT evidence is the primary source.

2. If the patient report contains:
   - a measurement
   - a test result
   - a finding
   - a pathology result
   - an imaging finding
   - a biomarker
   - a mutation
   - a value

   use the patient's actual result.

3. Do NOT replace a patient value with a NICE
   guideline value.

4. CORE/NICE evidence is secondary context.

5. Use CORE/NICE evidence to explain the medical
   significance of the patient's result when supported.

6. If a specific patient value is NOT found:
   explicitly say that the value was not found
   in the uploaded report.

7. NEVER invent, estimate, or infer a missing
   patient value.

8. Never use information from another patient's report.

9. Patient evidence is session/document scoped only.

10. If patient evidence conflicts with generic
    guideline context about the patient's actual
    measurement, the patient's report is the source
    of truth for the patient's value.

============================================================
"""

        grounded_prompt = (
            grounded_prompt
            + "\n\n"
            + patient_priority_instruction
        )

    # ========================================================
    # 9. LLM GENERATION
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "8. LLM GENERATION"
    )

    print(
        "=" * 75
    )

    try:

        generation_result = generate_answer(
            grounded_prompt=grounded_prompt
        )

    except Exception as error:

        if is_llm_quota_error(
            error
        ):

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
                source_mode=source_mode,
            )

            return build_success_result(
                stage="generation_fallback",
                reason="llm_quota_exhausted",
                query=query,
                session_id=session_id,
                source_mode=source_mode,
                safety_result=safety_result,
                evidence_result=evidence_result,
                core_results=core_results,
                patient_results=patient_results,
                generation_results=generation_results,
                citations=citations,
                grounded_prompt=grounded_prompt,
                answer=fallback_answer,
                generation={
                    "mode":
                        "grounded_fallback",

                    "llm_available":
                        False,

                    "error":
                        str(error),
                },
            )

        raise

    # ========================================================
    # 10. FINAL ANSWER VALIDATION
    # ========================================================

    answer = None

    if isinstance(
        generation_result,
        dict,
    ):

        answer = generation_result.get(
            "answer"
        )

    elif isinstance(
        generation_result,
        str,
    ):

        answer = generation_result

    # --------------------------------------------------------
    # Empty LLM answer
    # --------------------------------------------------------

    if (
        not answer
        or not str(answer).strip()
    ):

        print(
            "\nWARNING: "
            "LLM returned an empty answer."
        )

        fallback_answer = build_fallback_answer(
            query=query,
            retrieved_results=generation_results,
            citations=citations,
            source_mode=source_mode,
        )

        return build_success_result(
            stage="generation_fallback",
            reason="empty_llm_answer",
            query=query,
            session_id=session_id,
            source_mode=source_mode,
            safety_result=safety_result,
            evidence_result=evidence_result,
            core_results=core_results,
            patient_results=patient_results,
            generation_results=generation_results,
            citations=citations,
            grounded_prompt=grounded_prompt,
            answer=fallback_answer,
            generation=generation_result,
        )

    # ========================================================
    # FINAL SUCCESS
    # ========================================================

    return build_success_result(
        stage="generation",
        query=query,
        session_id=session_id,
        source_mode=source_mode,
        safety_result=safety_result,
        evidence_result=evidence_result,
        core_results=core_results,
        patient_results=patient_results,
        generation_results=generation_results,
        citations=citations,
        grounded_prompt=grounded_prompt,
        answer=str(answer).strip(),
        generation=generation_result,
    )


# ============================================================
# PRINT RESULT
# ============================================================

def print_pipeline_result(
    result: Dict[str, Any],
) -> None:

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
        result.get(
            "status"
        ),
    )

    print(
        "Stage:",
        result.get(
            "stage"
        ),
    )

    print(
        "Source Mode:",
        result.get(
            "source_mode"
        ),
    )

    print(
        "Session ID:",
        result.get(
            "session_id"
        ),
    )

    if result.get(
        "reason"
    ):

        print(
            "Reason:",
            result.get(
                "reason"
            ),
        )

    # ========================================================
    # CORE RESULTS
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
        [],
    )

    if not core_results:

        print(
            "No Core results."
        )

    for item in core_results:

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
                "",
            ),
            "|",
            metadata.get(
                "page_start",
                metadata.get(
                    "page",
                    "",
                ),
            ),
            "| score:",
            item.get(
                "hybrid_score",
                "",
            ),
        )

        print(
            "Content:"
        )

        print(
            item.get(
                "text",
                "",
            )
        )

    # ========================================================
    # PATIENT RESULTS
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
        [],
    )

    if not patient_results:

        print(
            "No Patient results."
        )

    for item in patient_results:

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
                "",
            ),
            "|",
            metadata.get(
                "page_start",
                metadata.get(
                    "page",
                    "",
                ),
            ),
            "| score:",
            item.get(
                "hybrid_score",
                "",
            ),
        )

        print(
            "Content:"
        )

        print(
            item.get(
                "text",
                "",
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
        [],
    )

    if not citations:

        print(
            "No citations."
        )

    for citation in citations:

        print(
            citation
        )

    # ========================================================
    # FINAL ANSWER
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
            "",
        )
    )

    print(
        "\n"
        + "=" * 75
    )


# ============================================================
# MANUAL TESTS
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 75
    )

    print(
        "PULMO GUIDE - "
        "END-TO-END PIPELINE TEST"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # TEST 1 — CORE
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

    try:

        result = run_pipeline(
            query=test_query,
            patient_pdf=None,
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
            str(error),
        )

    # ========================================================
    # TEST 2 — SEMANTIC CORE PARAPHRASE
    # ========================================================
    #
    # This test is important.
    #
    # It does NOT rely on the exact phrase
    # "lung cancer treatment".
    #
    # The semantic scope layer should help recognize
    # the lung-cancer domain.
    # ========================================================

    semantic_core_query = (
        "How should doctors manage non-small cell "
        "lung malignancy when treatment is intended "
        "to cure the disease?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 2 - HYBRID / SEMANTIC CORE"
    )

    print(
        "=" * 75
    )

    try:

        result = run_pipeline(
            query=semantic_core_query,
            patient_pdf=None,
        )

        print_pipeline_result(
            result
        )

    except Exception as error:

        print(
            "\nSEMANTIC CORE TEST ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            str(error),
        )

    # ========================================================
    # PATIENT PDF
    # ========================================================

    patient_pdf = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "patient"
        / "pulmonary_function_report.pdf"
    )

    session_id = (
        "test_patient_001"
    )

    # ========================================================
    # TEST 3 — PATIENT VALUE
    # ========================================================

    patient_query_1 = (
        "What is my FEV1?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 3 - PATIENT VALUE"
    )

    print(
        "=" * 75
    )

    if patient_pdf.exists():

        try:

            result = run_pipeline(
                query=patient_query_1,
                patient_pdf=str(
                    patient_pdf
                ),
                session_id=session_id,
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
                str(error),
            )

    else:

        print(
            "\nPatient PDF not found:",
            patient_pdf,
        )

    # ========================================================
    # TEST 4 — PATIENT FOLLOW-UP
    # ========================================================

    patient_query_2 = (
        "What does this result mean?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 4 - PATIENT FOLLOW-UP"
    )

    print(
        "=" * 75
    )

    if patient_pdf.exists():

        try:

            result = run_pipeline(
                query=patient_query_2,
                patient_pdf=str(
                    patient_pdf
                ),
                session_id=session_id,
            )

            print_pipeline_result(
                result
            )

        except Exception as error:

            print(
                "\nPATIENT FOLLOW-UP ERROR:"
            )

            print(
                type(error).__name__,
                ":",
                str(error),
            )

    # ========================================================
    # TEST 5 — PATIENT REPORT WITHOUT "MY"
    # ========================================================

    patient_query_3 = (
        "Is this finding concerning?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 5 - PATIENT FINDING"
    )

    print(
        "=" * 75
    )

    if patient_pdf.exists():

        try:

            result = run_pipeline(
                query=patient_query_3,
                patient_pdf=str(
                    patient_pdf
                ),
                session_id=session_id,
            )

            print_pipeline_result(
                result
            )

        except Exception as error:

            print(
                "\nPATIENT FINDING ERROR:"
            )

            print(
                type(error).__name__,
                ":",
                str(error),
            )

    # ========================================================
    # TEST 6 — OUT OF SCOPE: OTHER CANCER
    # ========================================================

    out_of_scope_query = (
        "What is the recommended treatment "
        "for pancreatic cancer according "
        "to this guideline?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 6 - OUT OF SCOPE / OTHER CANCER"
    )

    print(
        "=" * 75
    )

    try:

        result = run_pipeline(
            query=out_of_scope_query,
            patient_pdf=None,
        )

        print_pipeline_result(
            result
        )

    except Exception as error:

        print(
            "\nOUT-OF-SCOPE TEST ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            str(error),
        )

    # ========================================================
    # TEST 7 — OUT OF SCOPE: OTHER DISEASE
    # ========================================================

    rheumatoid_query = (
        "What are the symptoms of rheumatoid arthritis?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 7 - OUT OF SCOPE / OTHER DISEASE"
    )

    print(
        "=" * 75
    )

    try:

        result = run_pipeline(
            query=rheumatoid_query,
            patient_pdf=None,
        )

        print_pipeline_result(
            result
        )

    except Exception as error:

        print(
            "\nRHEUMATOID TEST ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            str(error),
        )

    # ========================================================
    # TEST 8 — OUT OF SCOPE: NON-MEDICAL
    # ========================================================

    non_medical_query = (
        "What is Shakira's latest album?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 8 - OUT OF SCOPE / NON-MEDICAL"
    )

    print(
        "=" * 75
    )

    try:

        result = run_pipeline(
            query=non_medical_query,
            patient_pdf=None,
        )

        print_pipeline_result(
            result
        )

    except Exception as error:

        print(
            "\nNON-MEDICAL TEST ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            str(error),
        )

    # ========================================================
    # TEST 9 — GENERIC QUESTION WITHOUT DOMAIN
    # ========================================================

    generic_query = (
        "What are the symptoms?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 9 - GENERIC QUESTION"
    )

    print(
        "=" * 75
    )

    try:

        result = run_pipeline(
            query=generic_query,
            patient_pdf=None,
        )

        print_pipeline_result(
            result
        )

    except Exception as error:

        print(
            "\nGENERIC QUESTION TEST ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            str(error),
        )

    # ========================================================
    # TEST 10 — GENERIC "MY" WITH PATIENT PDF
    # ========================================================

    generic_my_query = (
        "What is my favorite color?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 10 - GENERIC MY QUESTION"
    )

    print(
        "=" * 75
    )

    if patient_pdf.exists():

        try:

            result = run_pipeline(
                query=generic_my_query,
                patient_pdf=str(
                    patient_pdf
                ),
                session_id=session_id,
            )

            print_pipeline_result(
                result
            )

        except Exception as error:

            print(
                "\nGENERIC MY TEST ERROR:"
            )

            print(
                type(error).__name__,
                ":",
                str(error),
            )

    # ========================================================
    # TEST 11 — PATIENT PARAPHRASE
    # ========================================================

    patient_paraphrase_query = (
        "Can you explain the lung function "
        "measurement in the medical report I uploaded?"
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TEST 11 - HYBRID / PATIENT PARAPHRASE"
    )

    print(
        "=" * 75
    )

    if patient_pdf.exists():

        try:

            result = run_pipeline(
                query=patient_paraphrase_query,
                patient_pdf=str(
                    patient_pdf
                ),
                session_id=session_id,
            )

            print_pipeline_result(
                result
            )

        except Exception as error:

            print(
                "\nPATIENT PARAPHRASE ERROR:"
            )

            print(
                type(error).__name__,
                ":",
                str(error),
            )

    # ========================================================
    # TESTS COMPLETED
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TESTS COMPLETED"
    )

    print(
        "=" * 75
    )
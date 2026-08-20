"""
============================================================
PULMO GUIDE
STREAMLIT USER INTERFACE
============================================================

Purpose:
    User-facing interface for the Pulmo Guide RAG system.

Features:
    - Ask general lung-cancer questions
    - Upload a patient medical report
    - Ask patient-specific questions
    - Display grounded answers
    - Display citations
    - Display refusal messages safely

Backend:
    scripts/03_generation/pipeline.py

Run:
    streamlit run app.py
============================================================
"""

from pathlib import Path
import sys
import tempfile

import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

GENERATION_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "03_generation"
)

if str(GENERATION_DIR) not in sys.path:
    sys.path.append(str(GENERATION_DIR))


# ============================================================
# IMPORT PIPELINE
# ============================================================

from pipeline import run_pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pulmo Guide",
    page_icon="💝",
    layout="centered",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Medical color palette: teal / soft blue, calm and clinical */
    :root {
        --pulmo-primary: #0f766e;      /* deep teal */
        --pulmo-primary-light: #14b8a6;/* teal accent */
        --pulmo-secondary: #2563eb;    /* clinical blue */
        --pulmo-bg-soft: #ecfeff;      /* pale cyan background */
        --pulmo-bg-soft-2: #eff6ff;    /* pale blue background */
        --pulmo-warning-bg: #fff7ed;
        --pulmo-warning-border: #fb923c;
        --pulmo-text: #0f172a;
    }

    .stApp {
        background: linear-gradient(180deg, #f0fdfa 0%, #ffffff 320px);
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
        background: linear-gradient(90deg, var(--pulmo-primary), var(--pulmo-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
        color: #475569;
    }

    .source-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        background-color: var(--pulmo-bg-soft-2);
        border: 1px solid #bfdbfe;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        background-color: var(--pulmo-warning-bg);
        border: 1px solid var(--pulmo-warning-border);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--pulmo-bg-soft);
        border-right: 1px solid #99f6e4;
    }

    section[data-testid="stSidebar"] h2 {
        color: var(--pulmo-primary);
    }

    /* Buttons */
    .stButton>button, .stFileUploader button, div[data-testid="stChatInput"] button {
        background-color: var(--pulmo-primary) !important;
        color: white !important;
        border: none !important;
    }

    .stButton>button:hover, .stFileUploader button:hover {
        background-color: var(--pulmo-primary-light) !important;
    }

    /* Divider color */
    hr {
        border-color: #99f6e4 !important;
    }

    /* Subheaders */
    h3 {
        color: var(--pulmo-primary);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💝 Pulmo Guide</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered assistant for lung cancer information,
    grounded in trusted medical evidence.
    </div>
    """,
    unsafe_allow_html=True,
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About Pulmo Guide")

    st.write(
        """
        Pulmo Guide helps users understand information
        related to lung cancer using grounded medical
        evidence.
        """
    )

    st.info(
        """
        **Core Knowledge**

        NICE Lung Cancer guideline NG122.
        """
    )

    st.caption(
        "Pulmo Guide provides information for educational "
        "purposes and does not replace professional medical advice."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "patient_pdf_path" not in st.session_state:
    st.session_state.patient_pdf_path = None


if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = uuid.uuid4().hex


# ============================================================
# PATIENT REPORT UPLOAD
# ============================================================

st.subheader("📄 Patient Report")

uploaded_file = st.file_uploader(
    "Upload your medical report (PDF)",
    type=["pdf"],
    help=(
        "Upload a patient report if you want to ask "
        "questions about your own results."
    ),
)


if uploaded_file is not None:

    # Avoid rewriting the file if the same file is already loaded
    uploaded_identifier = (
        f"{uploaded_file.name}_{uploaded_file.size}"
    )

    if (
        st.session_state.get(
            "uploaded_identifier"
        )
        != uploaded_identifier
    ):

        temp_dir = Path(
            tempfile.gettempdir()
        ) / "pulmo_guide"

        temp_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        patient_path = (
            temp_dir
            / uploaded_file.name
        )

        with open(
            patient_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        st.session_state.patient_pdf_path = (
            str(patient_path)
        )

        st.session_state.uploaded_identifier = (
            uploaded_identifier
        )

        st.success(
            f"Report uploaded: {uploaded_file.name}"
        )

    else:

        st.success(
            f"Report ready: {uploaded_file.name}"
        )


else:

    st.session_state.patient_pdf_path = None


st.divider()


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("citations"):

            st.markdown(
                "**Sources**"
            )

            for citation in message["citations"]:

                st.markdown(
                    f"- {citation}"
                )


# ============================================================
# USER INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about lung cancer..."
)


# ============================================================
# PROCESS QUERY
# ============================================================

if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------------------
    # ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching medical evidence..."
        ):

            try:

                result = run_pipeline(

                    query=query,

                    patient_pdf=(
                        st.session_state.patient_pdf_path
                    ),

                    session_id=(
                        st.session_state.session_id
                        if st.session_state.patient_pdf_path
                        else None
                    ),
                )


                # ====================================================
                # RESULT STATUS
                # ====================================================

                status = result.get(
                    "status"
                )

                answer = result.get(
                    "answer",
                    ""
                )

                citations = result.get(
                    "citations",
                    []
                )


                # ====================================================
                # DISPLAY ANSWER
                # ====================================================

                if status == "refused":

                    st.warning(
                        answer
                    )

                else:

                    st.markdown(
                        answer
                    )


                # ====================================================
                # DISPLAY SOURCES
                # ====================================================

                if citations:

                    st.markdown(
                        "### Sources"
                    )

                    for citation in citations:

                        st.markdown(
                            f"- {citation}"
                        )


                # ====================================================
                # SAVE ASSISTANT MESSAGE
                # ====================================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                )


            except Exception as error:

                error_message = (
                    "Sorry, something went wrong "
                    "while processing your question."
                )

                st.error(
                    error_message
                )

                # Keep technical details out of
                # the user-facing interface.

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "citations": [],
                    }
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Pulmo Guide • Grounded Medical AI Assistant"
)

st.caption(
    "Information provided is for educational purposes "
    "and should not replace professional medical advice."
)
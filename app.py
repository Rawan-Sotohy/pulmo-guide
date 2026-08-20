"""
Pulmo Guide — Enterprise Healthcare AI Assistant
=================================================
Production-ready Streamlit front end.

Design language: clean "Healthcare SaaS" — teal primary, slate neutrals,
crisp white cards, calm greens for status. Modeled after modern dashboard
products (Vercel / Linear / Stripe) but softened for a clinical context.

Notes for integrators:
- Wire your real model call into `generate_response()`.
- The CSS block is intentionally aggressive about specificity because
  Streamlit's default widget markup (BaseWeb components) renders some
  elements — especially the selectbox dropdown menu — in a portal that
  sits OUTSIDE the sidebar's DOM subtree. That's why a naive
  `section[data-testid="stSidebar"] * { color: white }` rule leaves the
  open dropdown menu unreadable: the menu isn't a sidebar descendant.
  We fix this by styling `[data-baseweb="popover"]` / `[data-baseweb="menu"]`
  globally, not just inside the sidebar.
"""

import streamlit as st
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# EMBEDDED THEME CONFIG
# --------------------------------------------------------------------------
# Streamlit only reads its base theme (light/dark) from a physical
# .streamlit/config.toml file — it cannot be set via st.set_page_config().
# Forcing base="light" here matters: several widgets (most importantly the
# st.selectbox dropdown menu) render in a floating portal that inherits
# Streamlit's *default* theme rather than our CSS overrides. Without this,
# that portal falls back to a dark box even on a light page/sidebar.
# We write the file at runtime so everything ships from this single
# app.py — no separate config file to keep track of.
_CONFIG_DIR = Path(__file__).parent / ".streamlit"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"
_CONFIG_CONTENT = """\
[theme]
base = "light"
primaryColor = "#0F766E"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#0F172A"
font = "sans serif"

[server]
enableXsrfProtection = true
maxUploadSize = 25

[browser]
gatherUsageStats = false
"""
_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
if not _CONFIG_FILE.exists() or _CONFIG_FILE.read_text() != _CONFIG_CONTENT:
    _CONFIG_FILE.write_text(_CONFIG_CONTENT)

# --------------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit call
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Pulmo Guide | Pulmonology AI Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# DESIGN TOKENS
# --------------------------------------------------------------------------
TEAL = "#0F766E"        # primary brand
TEAL_DARK = "#0B5A54"
TEAL_SOFT = "#E6F4F2"    # tint backgrounds
GREEN = "#15803D"        # success / stable status
GREEN_SOFT = "#EAF7EE"
AMBER = "#B45309"        # caution status
AMBER_SOFT = "#FEF3E2"
RED = "#B91C1C"          # alert status
RED_SOFT = "#FDECEC"
SLATE_900 = "#0F172A"    # primary text
SLATE_600 = "#475569"    # secondary text
SLATE_400 = "#94A3B8"    # placeholder / muted
SLATE_200 = "#E2E8F0"    # borders
SLATE_50 = "#F8FAFC"     # app background
WHITE = "#FFFFFF"

# --------------------------------------------------------------------------
# GLOBAL CSS INJECTION
# --------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* ============================================================
           0. FONT + BASE RESET
           ============================================================ */
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                         Roboto, Helvetica, Arial, sans-serif;
        }}

        /* ============================================================
           1. APP BACKGROUND + DEFAULT TEXT COLOR (fixes low-contrast text)
           ============================================================ */
        .stApp {{
            background-color: {SLATE_50};
            color: {SLATE_900};
        }}

        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: {SLATE_50};
        }}

        [data-testid="stHeader"] {{
            background-color: transparent;
        }}

        /* All generic text-bearing elements inherit dark, readable ink */
        .stApp, .stApp p, .stApp span, .stApp li, .stApp label,
        .stMarkdown, .stMarkdown p, .stMarkdown li,
        h1, h2, h3, h4, h5, h6 {{
            color: {SLATE_900};
        }}

        .stApp .stCaption, [data-testid="stCaptionContainer"] {{
            color: {SLATE_600} !important;
        }}

        /* Main content max width + breathing room */
        .block-container {{
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 2.5rem;
            padding-right: 2.5rem;
        }}

        /* ============================================================
           2. SIDEBAR — background, text, spacing
           ============================================================ */
        section[data-testid="stSidebar"] {{
            background-color: {WHITE};
            border-right: 1px solid {SLATE_200};
        }}

        section[data-testid="stSidebar"] > div {{
            padding: 1.5rem 1.25rem 2rem 1.25rem;
        }}

        /* Sidebar text: force dark, legible ink (fixes invisible/clashing text) */
        section[data-testid="stSidebar"] * {{
            color: {SLATE_900};
        }}
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {{
            color: {SLATE_600} !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: {SLATE_200};
            margin: 1.1rem 0;
        }}

        /* Sidebar section headers */
        .sidebar-eyebrow {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {SLATE_400} !important;
            margin: 0.25rem 0 0.5rem 0;
        }}

        /* ============================================================
           3. INPUTS — text input, textarea, number input
           ============================================================ */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            background-color: {WHITE} !important;
            color: {SLATE_900} !important;
            border: 1px solid {SLATE_200} !important;
            border-radius: 8px !important;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus,
        .stNumberInput input:focus {{
            border-color: {TEAL} !important;
            box-shadow: 0 0 0 3px {TEAL_SOFT} !important;
        }}
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
            color: {SLATE_400} !important;
        }}

        /* ============================================================
           4. SELECTBOX / MULTISELECT — fixes "dark invisible box" bug
           ============================================================ */
        /* The closed control, wherever it lives (main area or sidebar) */
        div[data-baseweb="select"] > div {{
            background-color: {WHITE} !important;
            border: 1px solid {SLATE_200} !important;
            border-radius: 8px !important;
            color: {SLATE_900} !important;
        }}
        div[data-baseweb="select"] * {{
            color: {SLATE_900} !important;
        }}
        div[data-baseweb="select"] svg {{
            fill: {SLATE_600} !important;
        }}

        /* The OPEN dropdown menu is rendered in a portal appended to
           <body>, outside both .stApp and the sidebar — style it globally. */
        div[data-baseweb="popover"] {{
            z-index: 9999 !important;
        }}
        div[data-baseweb="popover"] ul[role="listbox"],
        div[data-baseweb="menu"],
        ul[data-testid="stSelectboxVirtualDropdown"] {{
            background-color: {WHITE} !important;
            border: 1px solid {SLATE_200} !important;
            border-radius: 8px !important;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12) !important;
        }}
        div[data-baseweb="popover"] li,
        ul[data-testid="stSelectboxVirtualDropdown"] li {{
            background-color: {WHITE} !important;
            color: {SLATE_900} !important;
        }}
        div[data-baseweb="popover"] li:hover,
        ul[data-testid="stSelectboxVirtualDropdown"] li:hover {{
            background-color: {TEAL_SOFT} !important;
            color: {TEAL_DARK} !important;
        }}
        div[data-baseweb="popover"] li[aria-selected="true"] {{
            background-color: {TEAL_SOFT} !important;
            color: {TEAL_DARK} !important;
            font-weight: 600 !important;
        }}

        /* Multiselect chips */
        span[data-baseweb="tag"] {{
            background-color: {TEAL} !important;
            color: {WHITE} !important;
            border-radius: 6px !important;
        }}

        /* ============================================================
           5. BUTTONS
           ============================================================ */
        .stButton > button, .stDownloadButton > button {{
            background-color: {TEAL};
            color: {WHITE} !important;
            border: 1px solid {TEAL};
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.1rem;
            transition: background-color 0.15s ease, box-shadow 0.15s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: {TEAL_DARK};
            border-color: {TEAL_DARK};
            box-shadow: 0 2px 8px rgba(15, 118, 110, 0.25);
        }}
        .stButton > button:focus {{
            box-shadow: 0 0 0 3px {TEAL_SOFT} !important;
        }}
        .stButton > button p {{
            color: {WHITE} !important;
        }}

        /* Secondary / ghost button variant via kind="secondary" */
        button[kind="secondary"] {{
            background-color: {WHITE} !important;
            color: {SLATE_900} !important;
            border: 1px solid {SLATE_200} !important;
        }}
        button[kind="secondary"] p {{
            color: {SLATE_900} !important;
        }}
        button[kind="secondary"]:hover {{
            border-color: {TEAL} !important;
            color: {TEAL_DARK} !important;
        }}

        /* ============================================================
           6. FILE UPLOADER
           ============================================================ */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {SLATE_50} !important;
            border: 1.5px dashed {SLATE_200} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {SLATE_600} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: {WHITE} !important;
            color: {SLATE_900} !important;
            border: 1px solid {SLATE_200} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button p {{
            color: {SLATE_900} !important;
        }}
        [data-testid="stFileUploaderFile"] {{
            background-color: {WHITE} !important;
            border: 1px solid {SLATE_200} !important;
            border-radius: 8px !important;
            color: {SLATE_900} !important;
        }}

        /* ============================================================
           7. RADIO / CHECKBOX / SLIDER labels
           ============================================================ */
        .stRadio label, .stCheckbox label {{
            color: {SLATE_900} !important;
        }}
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] {{
            color: {SLATE_600} !important;
        }}

        /* ============================================================
           8. EXPANDER
           ============================================================ */
        [data-testid="stExpander"] {{
            background-color: {WHITE};
            border: 1px solid {SLATE_200} !important;
            border-radius: 10px !important;
            overflow: hidden;
        }}
        [data-testid="stExpander"] summary {{
            background-color: {WHITE};
            color: {SLATE_900} !important;
            font-weight: 600;
            padding: 0.75rem 1rem !important;
        }}
        [data-testid="stExpander"] summary:hover {{
            background-color: {SLATE_50};
        }}
        [data-testid="stExpander"] summary svg {{
            fill: {SLATE_600} !important;
        }}
        [data-testid="stExpanderDetails"] {{
            background-color: {WHITE};
            padding: 0.25rem 1rem 1rem 1rem !important;
        }}

        /* ============================================================
           9. ALERTS (info / success / warning / error)
           ============================================================ */
        [data-testid="stAlertContentInfo"] {{ color: {TEAL_DARK} !important; }}
        [data-testid="stAlertContentSuccess"] {{ color: {GREEN} !important; }}
        [data-testid="stAlertContentWarning"] {{ color: {AMBER} !important; }}
        [data-testid="stAlertContentError"] {{ color: {RED} !important; }}
        div[data-baseweb="notification"] {{
            border-radius: 10px !important;
        }}

        /* ============================================================
           10. CHAT INTERFACE
           ============================================================ */
        [data-testid="stChatMessage"] {{
            background-color: {WHITE};
            border: 1px solid {SLATE_200};
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{
            color: {SLATE_900} !important;
        }}
        [data-testid="stChatMessageAvatarUser"] {{
            background-color: {SLATE_900} !important;
        }}
        [data-testid="stChatMessageAvatarAssistant"] {{
            background-color: {TEAL} !important;
        }}

        [data-testid="stChatInput"] {{
            background-color: {WHITE};
            border: 1px solid {SLATE_200};
            border-radius: 12px;
        }}
        [data-testid="stChatInput"] textarea {{
            color: {SLATE_900} !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {SLATE_400} !important;
        }}

        /* ============================================================
           11. CUSTOM COMPONENTS — banner, status cards, chips, footer
           ============================================================ */
        .pg-banner {{
            background: linear-gradient(135deg, {TEAL} 0%, {TEAL_DARK} 100%);
            border-radius: 16px;
            padding: 2rem 2.25rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(15, 118, 110, 0.18);
        }}
        .pg-banner h1 {{
            color: {WHITE} !important;
            font-size: 1.6rem;
            font-weight: 700;
            margin: 0 0 0.4rem 0;
        }}
        .pg-banner p {{
            color: rgba(255,255,255,0.88) !important;
            font-size: 0.95rem;
            margin: 0;
            max-width: 640px;
        }}
        .pg-banner-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.16);
            color: {WHITE};
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            margin-bottom: 0.75rem;
        }}

        .pg-card {{
            background-color: {WHITE};
            border: 1px solid {SLATE_200};
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            height: 100%;
        }}
        .pg-card-label {{
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            color: {SLATE_600};
            margin-bottom: 0.35rem;
        }}
        .pg-card-value {{
            font-size: 1.35rem;
            font-weight: 700;
            color: {SLATE_900};
            margin-bottom: 0.2rem;
        }}
        .pg-card-sub {{
            font-size: 0.82rem;
            color: {SLATE_600};
        }}

        .pg-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
        }}
        .pg-pill-green {{ background-color: {GREEN_SOFT}; color: {GREEN}; }}
        .pg-pill-amber {{ background-color: {AMBER_SOFT}; color: {AMBER}; }}
        .pg-pill-red   {{ background-color: {RED_SOFT};   color: {RED}; }}
        .pg-pill-teal  {{ background-color: {TEAL_SOFT};  color: {TEAL_DARK}; }}

        .pg-footer {{
            color: {SLATE_400};
            font-size: 0.78rem;
            text-align: center;
            padding-top: 1rem;
        }}

        .pg-disclaimer {{
            font-size: 0.85rem;
            color: {SLATE_600};
            line-height: 1.55;
        }}

        /* Hide default Streamlit chrome for a cleaner product feel */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}

        /* Divider tone */
        hr {{ border-color: {SLATE_200}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello, I'm **Pulmo Guide** — a clinical decision-support "
                "assistant for pulmonology. Ask me about spirometry "
                "interpretation, imaging findings, guideline-based "
                "management, or paste a case summary to get started."
            ),
        }
    ]

if "patient_context" not in st.session_state:
    st.session_state.patient_context = ""


# --------------------------------------------------------------------------
# MOCK RESPONSE GENERATOR — replace with your real model / API call
# --------------------------------------------------------------------------
def generate_response(user_input: str, model_name: str) -> str:
    """
    Placeholder logic so the app is runnable standalone.
    Swap this out for a real call, e.g.:

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": user_input}],
        )
        return response.content[0].text
    """
    return (
        f"*(Demo response from `{model_name}`)*\n\n"
        f"I received your message: \u201c{user_input}\u201d.\n\n"
        "Connect this function to your clinical LLM backend to return "
        "real, guideline-grounded pulmonology guidance here."
    )


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;">
            <div style="font-size:1.6rem;line-height:1;">🫁</div>
            <div>
                <div style="font-weight:700;font-size:1.05rem;color:{SLATE_900};">Pulmo Guide</div>
                <div style="font-size:0.72rem;color:{SLATE_600};">Clinical AI Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-eyebrow">Session</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Model",
        options=["Pulmo Guide — Clinical v2", "Pulmo Guide — Fast", "Pulmo Guide — Research"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-eyebrow">Patient context</div>', unsafe_allow_html=True)
    st.session_state.patient_context = st.text_area(
        "Patient context",
        value=st.session_state.patient_context,
        placeholder="Age, relevant history, current medications, key symptoms…",
        height=110,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-eyebrow">Attach records</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Attach records",
        type=["pdf", "png", "jpg", "jpeg", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) attached")

    st.markdown("<hr/>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New chat", use_container_width=True, type="secondary"):
            st.session_state.messages = st.session_state.messages[:1]
            st.rerun()
    with col_b:
        st.download_button(
            "Export",
            data="\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages
            ),
            file_name=f"pulmo_guide_transcript_{datetime.now():%Y%m%d_%H%M}.txt",
            use_container_width=True,
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-eyebrow">Status</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="pg-pill pg-pill-green">● Model online</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="pg-footer" style="text-align:left;padding-top:1.5rem;">
            Pulmo Guide v2.4.0<br/>For licensed clinician use.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# MAIN — WELCOME BANNER
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="pg-banner">
        <div class="pg-banner-badge">Pulmonology · Clinical Decision Support</div>
        <h1>Welcome back, Doctor.</h1>
        <p>
            Pulmo Guide helps you interpret pulmonary function tests, review
            imaging findings, and cross-check management plans against current
            guidelines — in seconds, not minutes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# STATUS CARDS
# --------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
card_data = [
    ("Active session", model_choice.split("—")[-1].strip(), "Clinical-grade reasoning", "teal"),
    ("Records attached", str(len(uploaded_files) if uploaded_files else 0), "PDF, imaging, labs", "green"),
    ("Context provided", "Yes" if st.session_state.patient_context.strip() else "None yet", "Improves specificity", "amber" if not st.session_state.patient_context.strip() else "green"),
    ("Guideline base", "GOLD · GINA · ATS/ERS", "Updated quarterly", "teal"),
]
for col, (label, value, sub, tone) in zip([c1, c2, c3, c4], card_data):
    with col:
        st.markdown(
            f"""
            <div class="pg-card">
                <div class="pg-card-label">{label}</div>
                <div class="pg-card-value">{value}</div>
                <div class="pg-card-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# --------------------------------------------------------------------------
# SAFETY / DISCLAIMER EXPANDER
# --------------------------------------------------------------------------
with st.expander("⚠️  Clinical use disclaimer", expanded=False):
    st.markdown(
        """
        <div class="pg-disclaimer">
        Pulmo Guide is a decision-support tool intended for use by licensed
        healthcare professionals. It does not replace clinical judgment,
        direct patient evaluation, or institutional protocols. Always verify
        AI-generated suggestions against current guidelines and the
        patient's full clinical picture before acting on them. Do not enter
        directly identifiable patient information.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------------------------------
st.markdown("##### Consultation")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        avatar = "🫁" if msg["role"] == "assistant" else "🧑‍⚕️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

user_input = st.chat_input("Ask about a case, PFT results, imaging, or guideline-based management…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(user_input)

        full_prompt = user_input
        if st.session_state.patient_context.strip():
            full_prompt = f"Patient context: {st.session_state.patient_context}\n\nQuestion: {user_input}"

        with st.chat_message("assistant", avatar="🫁"):
            with st.spinner("Reviewing clinical guidelines…"):
                reply = generate_response(full_prompt, model_choice)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown(
    '<div class="pg-footer">Pulmo Guide is a decision-support tool, not a diagnostic device. '
    'Verify all outputs against clinical judgment.</div>',
    unsafe_allow_html=True,
)
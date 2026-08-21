<div align="center">

# 💝 Pulmo Guide

**is a medical RAG assistant designed to help users understand information related to **lung cancer** using a trusted medical knowledge source.**
</div>

---

## 🏆 About This Project

**Pulmo Guide** was built during **AI Hackathon Alexandria**, a hackathon focused on applying AI to real-world healthcare challenges. Our goal was to design a medical assistant that doesn't just *sound* confident — it stays grounded in trusted clinical evidence and knows when to say "I don't know."

## 🎯 What It Does

Pulmo Guide helps users understand information related to **lung cancer**, grounded in the **NICE Guideline NG122 — Lung cancer: diagnosis and management**. Users can:

- 💬 Ask general questions about lung cancer (symptoms, diagnosis, treatment pathways)
- 📄 Upload a personal medical report (PDF) and ask patient-specific questions
- ✅ Receive answers that are **evidence-grounded and cited**
- 🛑 Get a transparent refusal instead of a guess, whenever the evidence isn't strong enough

## 🧠 Why "Guarded" RAG?

Most RAG chatbots retrieve *something* and answer *anyway*. In a medical context, that's dangerous. Pulmo Guide adds two extra checkpoints most systems skip:

1. A **Safety Decision** layer that filters unsafe, out-of-scope, or injection-style queries *before* retrieval even happens.
2. An **Evidence Decision** layer that scores retrieval confidence and **refuses to answer** when the evidence is insufficient — instead of hallucinating a confident-sounding response.

---

### 🔄 End-to-End RAG Query Flow

```text
                 USER QUERY
                      │
                      ▼
             ┌──────────────────┐
             │ SAFETY DECISION  │
             └──────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
       UNSAFE    OUT_OF_SCOPE  INJECTION
          │           │           │
          └───────────┴───────────┘
                      │
                    REFUSE


                      │
                  IN_SCOPE
                      ▼
             ┌──────────────────┐
             │  SOURCE ROUTING  │
             └──────────────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ HYBRID RETRIEVAL │
             │     70 / 30      │
             │ Dense + BM25     │
             └──────────────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ EVIDENCE DECISION│
             └──────────────────┘
                      │
             ┌────────┴────────┐
             │                 │
        INSUFFICIENT        SUFFICIENT
             │                 │
           REFUSE              │
                               ▼
                    ┌──────────────────┐
                    │ GROUNDED PROMPT  │
                    │ Persona +        │
                    │ Evidence         │
                    └──────────────────┘
                               │
                               ▼
                         ┌──────────┐
                         │  GEMINI  │
                         │   LLM    │
                         └──────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ CITATION MAPPING │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ CITATION         │
                    │ VALIDATION       │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ FINAL RESPONSE   │
                    │ Answer +         │       
                    │ Citations +      │
                    │ Sources          │
                    └──────────────────┘
```

---

## 🧩 Core Components

| Stage | Role |
|---|---|
| **Safety Decision** | Filters unsafe, out-of-scope, or prompt-injection queries before any retrieval happens |
| **Hybrid Retrieval (70/30)** | Combines dense vector search with keyword/BM25 search for more robust recall |
| **Evidence Decision** | Scores retrieved context and decides: refuse, answer with caveats, or answer with confidence |
| **Grounded Prompt** | Injects persona instructions + evidence strength into the LLM prompt |
| **Citation Validation** | Verifies that every claim in the final answer traces back to a real source passage |

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Retrieval:** ChromaDB (vector search) + Hybrid(Semantic+Rank-bm25)
- **Embeddings:** BAAI/bge-small-en-v1.5
- **PDF Parsing:** Docling / PyMuPDF
- **Knowledge Base:** NICE Guideline NG122 (Lung Cancer)

---

## 🚀 Links

| Resource | Link |
|---|---|
| 🔴 Live Demo | [Open App](https://pulmo-guide-2026.streamlit.app/) |
| 🎥 Video Demo | [Watch the Video](https://drive.google.com/file/d/1ZmCUwkZLGs_9DIB0AMR0-yD2vW0UkGrE/view?usp=sharing) |
| 📊 Presentation | [View Slides](https://canva.link/hi12nnuvhrijw0n) |

---

<div align="center">

Made with 🩺 at **AI Hackathon Alexandria 2026**

</div>

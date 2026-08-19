# 🫁 Pulmo Guide

**Pulmo Guide** is a medical RAG assistant designed to help users understand information related to **lung cancer** using a trusted medical knowledge source.

The system is grounded in the **NICE guideline: Lung cancer: diagnosis and management (NG122)** and is designed to provide evidence-based answers while reducing unsupported or hallucinated responses.

---

## 🎯 Project Goal

Pulmo Guide follows a guarded RAG architecture:

```text
User Question
      ↓
Hybrid Retrieval
      ↓
Refusal / Confidence Gate
      ↓
Grounded Prompt
      ↓
LLM Generation
      ↓
Answer + Evidence + Citation
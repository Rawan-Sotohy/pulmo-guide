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
```


---


```

                         USER QUERY
                              │
                              ▼
                    ┌──────────────────┐
                    │ SAFETY DECISION  │
                    └──────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
           UNSAFE       OUT_OF_SCOPE    INJECTION
              │               │               │
              └───────────────┴───────────────┘
                              │
                           REFUSE
                             
                              │
                         IN_SCOPE
                              ▼
                  ┌─────────────────────┐
                  │ HYBRID RETRIEVAL    │
                  │      70 / 30        │
                  └─────────────────────┘
                              │
                              ▼
                  ┌─────────────────────┐
                  │ EVIDENCE DECISION   │
                  └─────────────────────┘
                              │
             ┌────────────────┼────────────────┐
             │                │                │
        insufficient        weak/partial     strong
             │                │                │
           REFUSE             │                │
                              └───────┬────────┘
                                      ▼
                           ┌──────────────────┐
                           │ GROUNDED PROMPT  │
                           │ Persona +        │
                           │ Evidence Level   │
                           └──────────────────┘
                                      │
                                      ▼
                              ┌─────────────┐
                              │     LLM     │
                              └─────────────┘
                                      │
                                      ▼
                       Recommendation / Excerpt / Citation
                                      │
                                      ▼
                              Citation Validation


 ```
# Pulmo Guide - Refusal Threshold Experiment

Dataset size: **30 questions**
- In-KB: **15**
- Out-of-KB: **15**

## Retrieval Configuration

- Embedding model: `BAAI/bge-small-en-v1.5`
- Semantic weight: **70%**
- BM25 weight: **30%**
- Alpha: `0.7`
- Final Top-K: `5`
- Reranker: **None**
- Confidence signal: **Rank-1 Hybrid score**

## Threshold Results

| Threshold | Accepted | Rejected | Correct Accept | Correct Refuse | False Accept | False Reject | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 30 | 0 | 15 | 0 | 15 | 0 | 0.5 | 1.0 | 0.5 |
| 0.55 | 30 | 0 | 15 | 0 | 15 | 0 | 0.5 | 1.0 | 0.5 |
| 0.6 | 30 | 0 | 15 | 0 | 15 | 0 | 0.5 | 1.0 | 0.5 |
| 0.65 | 30 | 0 | 15 | 0 | 15 | 0 | 0.5 | 1.0 | 0.5 |
| 0.7 | 30 | 0 | 15 | 0 | 15 | 0 | 0.5 | 1.0 | 0.5 |
| 0.75 | 30 | 0 | 15 | 0 | 15 | 0 | 0.5 | 1.0 | 0.5 |
| 0.8 | 28 | 2 | 15 | 2 | 13 | 0 | 0.5357 | 1.0 | 0.5667 |

## Interpretation

The Hybrid score is used as a retrieval-based signal for the refusal experiment. It should not be interpreted as a calibrated probability.

The final threshold should be selected based on the trade-off between false accepts and false rejects, with special attention to out-of-KB questions.

**Important:** This experiment is directional. A larger and more diverse labeled dataset is recommended before treating the threshold as fully validated.
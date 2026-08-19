# Pulmo Guide — Retrieval Evaluation Report

Evaluated **32 questions** from `evaluation_set.json` against **4 retrieval configurations**, cutoff k=5, ChromaDB collection `pulmo_guide`.

Total evaluation time: 31.7s

## Configuration

- Embedding model: `BAAI/bge-small-en-v1.5`
- Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Alpha (semantic weight): `0.7`
- Semantic weight: `70%`
- BM25 weight: `30%`
- Candidate K (pre-rerank): `10`
- Final Top K: `5`

## Comparison Table (averaged over all questions)

| Method | Precision@5 | Recall@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|
| Semantic Search | 0.338 | 0.870 | 0.938 | 0.860 |
| BM25 | 0.325 | 0.862 | 0.906 | 0.839 |
| Hybrid 70/30 | 0.356 | 0.922 | 0.969 | 0.891 |
| Hybrid + MS-MARCO | 0.344 | 0.891 | 0.938 | 0.875 |

## Per-Question Results

### 1. In what specific clinical scenario should sputum cytology be used for investigating suspected lung cancer?

**Ground truth:** `['core_0013', 'core_0014']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0014', 'core_0013', 'core_0050', 'core_0021', 'core_0033']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0013', 'core_0050', 'core_0049', 'core_0014', 'core_0010']` | `[True, False, False, True, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0013', 'core_0014', 'core_0050', 'core_0049', 'core_0033']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0013', 'core_0014', 'core_0050', 'core_0049', 'core_0029']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 2. What should be included in the contrast-enhanced chest CT scan offered to people with known or suspected lung cancer, and what precaution applies to people with renal impairment?

**Ground truth:** `['core_0014', 'core_0015']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0014', 'core_0015', 'core_0040', 'core_0041', 'core_0032']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0014', 'core_0015', 'core_0039', 'core_0040', 'core_0180']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0014', 'core_0015', 'core_0040', 'core_0039', 'core_0041']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0014', 'core_0015', 'core_0032', 'core_0040', 'core_0041']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 3. Who should be offered PET-CT scanning before treatment for lung cancer?

**Ground truth:** `['core_0014', 'core_0015']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0032', 'core_0015', 'core_0014', 'core_0031', 'core_0033']` | `[False, True, True, False, False]` | 2 | 0.40 | 1.00 | 1 | 0.50 |
| BM25 | `['core_0015', 'core_0016', 'core_0014', 'core_0032', 'core_0063']` | `[True, False, True, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0015', 'core_0032', 'core_0016', 'core_0014', 'core_0031']` | `[True, False, False, True, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0015', 'core_0032', 'core_0016', 'core_0031', 'core_0014']` | `[True, False, False, False, True]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 4. Should MRI be routinely used to assess the T-stage of the primary tumour in NSCLC, and when is MRI specifically indicated instead?

**Ground truth:** `['core_0016', 'core_0017']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0016', 'core_0017', 'core_0041', 'core_0038', 'core_0040']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0017', 'core_0016', 'core_0004', 'core_0003', 'core_0150']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0016', 'core_0017', 'core_0041', 'core_0038', 'core_0040']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0017', 'core_0016', 'core_0038', 'core_0041', 'core_0040']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 5. What is EBUS-TBNA used for in the diagnostic pathway for lung cancer, and why was clinical audit of EBUS-TBNA and EUS-FNA recommended by the committee?

**Ground truth:** `['core_0016', 'core_0017', 'core_0018']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0018', 'core_0019', 'core_0036', 'core_0022', 'core_0020']` | `[True, False, False, False, False]` | 1 | 0.20 | 0.33 | 1 | 1.00 |
| BM25 | `['core_0019', 'core_0034', 'core_0033', 'core_0018', 'core_0022']` | `[False, False, False, True, False]` | 4 | 0.20 | 0.33 | 1 | 0.25 |
| Hybrid 70/30 | `['core_0018', 'core_0019', 'core_0022', 'core_0034', 'core_0020']` | `[True, False, False, False, False]` | 1 | 0.20 | 0.33 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0019', 'core_0018', 'core_0023', 'core_0033', 'core_0034']` | `[False, True, False, False, False]` | 2 | 0.20 | 0.33 | 1 | 0.50 |

### 6. What guidance should be followed for next-generation sequencing (NGS) panel testing to guide lung cancer treatment, and what quality requirement applies to tissue samples taken for pathological diagnosis?

**Ground truth:** `['core_0018', 'core_0019']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0019', 'core_0208', 'core_0209', 'core_0006', 'core_0018']` | `[True, False, False, False, True]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0019', 'core_0018', 'core_0208', 'core_0180', 'core_0209']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0019', 'core_0208', 'core_0018', 'core_0209', 'core_0180']` | `[True, False, True, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0019', 'core_0018', 'core_0209', 'core_0208', 'core_0006']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 7. When choosing which lesion to biopsy in a patient with a peripheral primary tumour, should enlarged intrathoracic nodes or the primary lesion be prioritised, and under what size threshold are nodes considered enlarged?

**Ground truth:** `['core_0029']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0029', 'core_0033', 'core_0034', 'core_0031', 'core_0030']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0029', 'core_0033', 'core_0034', 'core_0032', 'core_0016']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0029', 'core_0033', 'core_0034', 'core_0032', 'core_0031']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0029', 'core_0017', 'core_0016', 'core_0030', 'core_0034']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |

### 8. When is flexible bronchoscopy recommended for people with a central lung lesion on CT?

**Ground truth:** `['core_0030']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0030', 'core_0014', 'core_0023', 'core_0026', 'core_0032']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0030', 'core_0022', 'core_0029', 'core_0036', 'core_0023']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0030', 'core_0014', 'core_0023', 'core_0022', 'core_0029']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0030', 'core_0014', 'core_0029', 'core_0016', 'core_0023']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |

### 9. What is the recommended staging pathway for intrathoracic lymph nodes in patients with a low probability of nodal malignancy versus those with enlarged (≥10 mm) intrathoracic lymph nodes who could have curative-intent treatment?

**Ground truth:** `['core_0031', 'core_0032', 'core_0033']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0033', 'core_0034', 'core_0031', 'core_0032', 'core_0023']` | `[True, False, True, True, False]` | 1 | 0.60 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0031', 'core_0032', 'core_0033', 'core_0034', 'core_0022']` | `[True, True, True, False, False]` | 1 | 0.60 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0033', 'core_0031', 'core_0032', 'core_0034', 'core_0022']` | `[True, True, True, False, False]` | 1 | 0.60 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0031', 'core_0032', 'core_0023', 'core_0034', 'core_0033']` | `[True, True, False, False, True]` | 1 | 0.60 | 1.00 | 1 | 1.00 |

### 10. When should surgical mediastinal staging be considered after EBUS-TBNA or EUS-FNA results?

**Ground truth:** `['core_0033', 'core_0034']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0022', 'core_0036', 'core_0018', 'core_0023', 'core_0035']` | `[False, False, False, False, False]` | None | 0.00 | 0.00 | 0 | 0.00 |
| BM25 | `['core_0034', 'core_0023', 'core_0022', 'core_0033', 'core_0035']` | `[True, False, False, True, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0022', 'core_0034', 'core_0023', 'core_0033', 'core_0035']` | `[False, True, False, True, False]` | 2 | 0.40 | 1.00 | 1 | 0.50 |
| Hybrid + MS-MARCO | `['core_0023', 'core_0034', 'core_0022', 'core_0033', 'core_0026']` | `[False, True, False, True, False]` | 2 | 0.40 | 1.00 | 1 | 0.50 |

### 11. What brain imaging is recommended for people with clinical stage 1, stage 2, and stage 3 NSCLC who are being considered for curative-intent treatment?

**Ground truth:** `['core_0038', 'core_0039', 'core_0040', 'core_0041']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0039', 'core_0038', 'core_0040', 'core_0043', 'core_0041']` | `[True, True, True, False, True]` | 1 | 0.80 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0039', 'core_0040', 'core_0043', 'core_0038', 'core_0085']` | `[True, True, False, True, False]` | 1 | 0.60 | 0.75 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0039', 'core_0038', 'core_0040', 'core_0043', 'core_0041']` | `[True, True, True, False, True]` | 1 | 0.80 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0040', 'core_0039', 'core_0041', 'core_0043', 'core_0038']` | `[True, True, True, False, True]` | 1 | 0.80 | 1.00 | 1 | 1.00 |

### 12. What is the recommended first-line imaging test for a patient with localised signs or symptoms of bone metastasis, and what should be done if the result is negative or inconclusive?

**Ground truth:** `['core_0041', 'core_0042']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0042', 'core_0041', 'core_0171', 'core_0038', 'core_0037']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0042', 'core_0041', 'core_0186', 'core_0043', 'core_0183']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0042', 'core_0041', 'core_0038', 'core_0043', 'core_0172']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0042', 'core_0041', 'core_0172', 'core_0171', 'core_0043']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 13. In what circumstances should bone scintigraphy be avoided in lung cancer staging?

**Ground truth:** `['core_0041', 'core_0042']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0042', 'core_0015', 'core_0022', 'core_0023', 'core_0014']` | `[True, False, False, False, False]` | 1 | 0.20 | 0.50 | 1 | 1.00 |
| BM25 | `['core_0042', 'core_0041', 'core_0172', 'core_0187', 'core_0003']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0042', 'core_0041', 'core_0015', 'core_0014', 'core_0022']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0042', 'core_0023', 'core_0041', 'core_0022', 'core_0014']` | `[True, False, True, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 14. What cardiovascular assessment criteria determine whether a person with NSCLC can proceed to surgery without further cardiac investigation versus needing a cardiology review?

**Ground truth:** `['core_0058', 'core_0059']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0057', 'core_0071', 'core_0070', 'core_0065', 'core_0059']` | `[False, False, False, False, True]` | 5 | 0.20 | 0.50 | 1 | 0.20 |
| BM25 | `['core_0058', 'core_0059', 'core_0063', 'core_0085', 'core_0090']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0058', 'core_0059', 'core_0057', 'core_0071', 'core_0070']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0058', 'core_0059', 'core_0057', 'core_0071', 'core_0085']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 15. What lung function tests should be performed before proceeding with curative-intent treatment for lung cancer, and what threshold is used for shuttle walk testing to assess fitness for surgery?

**Ground truth:** `['core_0062']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0062', 'core_0070', 'core_0071', 'core_0063', 'core_0050']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0062', 'core_0180', 'core_0070', 'core_0179', 'core_0071']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0062', 'core_0070', 'core_0071', 'core_0063', 'core_0180']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0062', 'core_0032', 'core_0070', 'core_0071', 'core_0063']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |

### 16. What type of surgery should be offered to people with NSCLC who are well enough for curative-intent treatment, and when is more extensive surgery such as pneumonectomy indicated?

**Ground truth:** `['core_0065', 'core_0066']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0066', 'core_0065', 'core_0067', 'core_0085', 'core_0084']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0066', 'core_0084', 'core_0085', 'core_0088', 'core_0065']` | `[True, False, False, False, True]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0066', 'core_0065', 'core_0084', 'core_0085', 'core_0067']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0066', 'core_0065', 'core_0084', 'core_0067', 'core_0085']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 17. What treatment options are recommended for people with stage 1 to 2a NSCLC who decline lobectomy or in whom it is contraindicated?

**Ground truth:** `['core_0069']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0069', 'core_0072', 'core_0076', 'core_0065', 'core_0071']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0069', 'core_0072', 'core_0076', 'core_0071', 'core_0065']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0069', 'core_0072', 'core_0076', 'core_0071', 'core_0065']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0069', 'core_0076', 'core_0072', 'core_0071', 'core_0065']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |

### 18. What are the two recommended conventionally fractionated radical radiotherapy dose regimens for NSCLC?

**Ground truth:** `['core_0074', 'core_0075']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0077', 'core_0073', 'core_0078', 'core_0072', 'core_0071']` | `[False, False, False, False, False]` | None | 0.00 | 0.00 | 0 | 0.00 |
| BM25 | `['core_0077', 'core_0078', 'core_0076', 'core_0074', 'core_0075']` | `[False, False, False, True, True]` | 4 | 0.40 | 1.00 | 1 | 0.25 |
| Hybrid 70/30 | `['core_0077', 'core_0078', 'core_0076', 'core_0075', 'core_0074']` | `[False, False, False, True, True]` | 4 | 0.40 | 1.00 | 1 | 0.25 |
| Hybrid + MS-MARCO | `['core_0077', 'core_0076', 'core_0072', 'core_0078', 'core_0071']` | `[False, False, False, False, False]` | None | 0.00 | 0.00 | 0 | 0.00 |

### 19. For people with operable stage 3a N2 NSCLC having chemoradiotherapy and surgery, how long after completing chemoradiotherapy should surgery be scheduled, and what benefits should be discussed with the person before starting this combined treatment?

**Ground truth:** `['core_0085', 'core_0086']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0086', 'core_0085', 'core_0089', 'core_0090', 'core_0091']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0085', 'core_0086', 'core_0089', 'core_0090', 'core_0084']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0085', 'core_0086', 'core_0089', 'core_0090', 'core_0091']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0085', 'core_0086', 'core_0089', 'core_0091', 'core_0090']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 20. For which patients with resectable NSCLC is nivolumab in combination with chemotherapy recommended as neoadjuvant treatment?

**Ground truth:** `['core_0093']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0093', 'core_0095', 'core_0094', 'core_0090', 'core_0180']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0093', 'core_0095', 'core_0094', 'core_0129', 'core_0098']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0093', 'core_0095', 'core_0094', 'core_0129', 'core_0090']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0093', 'core_0095', 'core_0094', 'core_0098', 'core_0129']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |

### 21. For which patients with resected NSCLC is osimertinib recommended as adjuvant treatment, and when should it be stopped?

**Ground truth:** `['core_0098', 'core_0099']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0099', 'core_0098', 'core_0100', 'core_0101', 'core_0093']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0099', 'core_0098', 'core_0100', 'core_0101', 'core_0045']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0099', 'core_0098', 'core_0100', 'core_0101', 'core_0094']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0099', 'core_0098', 'core_0100', 'core_0101', 'core_0095']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 22. For which patients with resected NSCLC is alectinib recommended as adjuvant treatment?

**Ground truth:** `['core_0100', 'core_0101']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0101', 'core_0100', 'core_0095', 'core_0094', 'core_0099']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0101', 'core_0100', 'core_0098', 'core_0099', 'core_0094']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0101', 'core_0100', 'core_0098', 'core_0099', 'core_0094']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0101', 'core_0100', 'core_0093', 'core_0099', 'core_0094']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 23. What first-line chemotherapy regimen and radiotherapy schedule are recommended for people with limited-stage SCLC?

**Ground truth:** `['core_0135', 'core_0136']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0136', 'core_0138', 'core_0145', 'core_0141', 'core_0135']` | `[True, False, False, False, True]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0154', 'core_0141', 'core_0138', 'core_0155', 'core_0139']` | `[False, False, False, False, False]` | None | 0.00 | 0.00 | 0 | 0.00 |
| Hybrid 70/30 | `['core_0138', 'core_0154', 'core_0141', 'core_0139', 'core_0155']` | `[False, False, False, False, False]` | None | 0.00 | 0.00 | 0 | 0.00 |
| Hybrid + MS-MARCO | `['core_0138', 'core_0141', 'core_0157', 'core_0139', 'core_0154']` | `[False, False, False, False, False]` | None | 0.00 | 0.00 | 0 | 0.00 |

### 24. What is the recommended dose and fractionation for prophylactic cranial irradiation in limited-stage SCLC, and under what condition should it be offered?

**Ground truth:** `['core_0138', 'core_0139']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0138', 'core_0139', 'core_0147', 'core_0189', 'core_0151']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0187', 'core_0188', 'core_0189', 'core_0151', 'core_0150']` | `[False, False, False, False, False]` | None | 0.00 | 0.00 | 0 | 0.00 |
| Hybrid 70/30 | `['core_0139', 'core_0138', 'core_0189', 'core_0187', 'core_0151']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0139', 'core_0138', 'core_0147', 'core_0146', 'core_0151']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 25. What chemotherapy is recommended as first-line treatment for extensive-stage SCLC, and what is the maximum number of cycles that should be offered?

**Ground truth:** `['core_0143', 'core_0144']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0145', 'core_0144', 'core_0154', 'core_0155', 'core_0146']` | `[False, True, False, False, False]` | 2 | 0.20 | 0.50 | 1 | 0.50 |
| BM25 | `['core_0145', 'core_0151', 'core_0144', 'core_0188', 'core_0187']` | `[False, False, True, False, False]` | 3 | 0.20 | 0.50 | 1 | 0.33 |
| Hybrid 70/30 | `['core_0145', 'core_0144', 'core_0154', 'core_0155', 'core_0157']` | `[False, True, False, False, False]` | 2 | 0.20 | 0.50 | 1 | 0.50 |
| Hybrid + MS-MARCO | `['core_0145', 'core_0144', 'core_0155', 'core_0154', 'core_0157']` | `[False, True, False, False, False]` | 2 | 0.20 | 0.50 | 1 | 0.50 |

### 26. What treatment options are available for people with SCLC that has relapsed after first-line treatment, and what should patients be told if their disease did not respond to first-line treatment?

**Ground truth:** `['core_0154', 'core_0155', 'core_0156']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0152', 'core_0153', 'core_0156', 'core_0154', 'core_0155']` | `[False, False, True, True, True]` | 3 | 0.60 | 1.00 | 1 | 0.33 |
| BM25 | `['core_0153', 'core_0152', 'core_0157', 'core_0138', 'core_0139']` | `[False, False, False, False, False]` | None | 0.00 | 0.00 | 0 | 0.00 |
| Hybrid 70/30 | `['core_0153', 'core_0152', 'core_0157', 'core_0156', 'core_0154']` | `[False, False, False, True, True]` | 4 | 0.40 | 0.67 | 1 | 0.25 |
| Hybrid + MS-MARCO | `['core_0157', 'core_0156', 'core_0152', 'core_0153', 'core_0154']` | `[False, True, False, False, True]` | 2 | 0.40 | 0.67 | 1 | 0.50 |

### 27. What interventions should be offered to people with lung cancer who have impending endobronchial obstruction?

**Ground truth:** `['core_0160', 'core_0161']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0161', 'core_0160', 'core_0053', 'core_0056', 'core_0165']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0161', 'core_0160', 'core_0164', 'core_0015', 'core_0165']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0161', 'core_0160', 'core_0165', 'core_0014', 'core_0056']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0161', 'core_0160', 'core_0015', 'core_0014', 'core_0056']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 28. What treatment options are available for people presenting with superior vena cava obstruction due to lung cancer?

**Ground truth:** `['core_0167', 'core_0168']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0168', 'core_0167', 'core_0033', 'core_0014', 'core_0034']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0168', 'core_0167', 'core_0160', 'core_0166', 'core_0016']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0168', 'core_0167', 'core_0160', 'core_0161', 'core_0033']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0168', 'core_0167', 'core_0126', 'core_0161', 'core_0125']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 29. What is the recommended management of dexamethasone dosing in people with symptomatic brain metastases from lung cancer?

**Ground truth:** `['core_0169', 'core_0170']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0169', 'core_0170', 'core_0188', 'core_0187', 'core_0145']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0170', 'core_0169', 'core_0043', 'core_0188', 'core_0187']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0170', 'core_0169', 'core_0188', 'core_0043', 'core_0187']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0170', 'core_0169', 'core_0172', 'core_0188', 'core_0043']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 30. When is denosumab recommended for preventing skeletal-related events in people with bone metastases from lung cancer?

**Ground truth:** `['core_0172', 'core_0173']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0173', 'core_0172', 'core_0180', 'core_0191', 'core_0129']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0172', 'core_0173', 'core_0171', 'core_0042', 'core_0041']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0173', 'core_0172', 'core_0171', 'core_0180', 'core_0129']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0173', 'core_0172', 'core_0171', 'core_0102', 'core_0140']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 31. Within how many weeks of completing treatment should people with lung cancer be offered an initial follow-up appointment, and who can lead protocol-driven follow-up?

**Ground truth:** `['core_0175', 'core_0176']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0175', 'core_0176', 'core_0177', 'core_0053', 'core_0131']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0176', 'core_0175', 'core_0177', 'core_0044', 'core_0015']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0176', 'core_0175', 'core_0177', 'core_0131', 'core_0053']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0176', 'core_0175', 'core_0177', 'core_0131', 'core_0046']` | `[True, True, False, False, False]` | 1 | 0.40 | 1.00 | 1 | 1.00 |

### 32. Should surgery for lung cancer be postponed to allow a person time to stop smoking first?

**Ground truth:** `['core_0056']`

| Method | Retrieved (top 5) | Relevant | First rel. rank | P@5 | R@5 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|---|---|
| Semantic Search | `['core_0056', 'core_0055', 'core_0054', 'core_0046', 'core_0079']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| BM25 | `['core_0056', 'core_0055', 'core_0054', 'core_0088', 'core_0089']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid 70/30 | `['core_0056', 'core_0055', 'core_0054', 'core_0088', 'core_0089']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |
| Hybrid + MS-MARCO | `['core_0056', 'core_0055', 'core_0054', 'core_0046', 'core_0079']` | `[True, False, False, False, False]` | 1 | 0.20 | 1.00 | 1 | 1.00 |

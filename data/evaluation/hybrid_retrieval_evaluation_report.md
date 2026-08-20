# Pulmo Guide — Hybrid Retrieval Evaluation

## Configuration

- Embedding model: `BAAI/bge-small-en-v1.5`
- Semantic weight: `70%`
- BM25 weight: `30%`
- Final Top K: `5`
- Reranker: `OFF`

## Overall Metrics

| Metric | Score |
|---|---:|
| Precision@5 | 0.356 |
| Recall@5 | 0.922 |
| Hit@5 | 0.969 |
| MRR@5 | 0.891 |

## Per-Question Results

### 1. In what specific clinical scenario should sputum cytology be used for investigating suspected lung cancer?

**Ground truth:** `['core_0013', 'core_0014']`

**Retrieved Top 5:** `['core_0013', 'core_0014', 'core_0050', 'core_0049', 'core_0033']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 2. What should be included in the contrast-enhanced chest CT scan offered to people with known or suspected lung cancer, and what precaution applies to people with renal impairment?

**Ground truth:** `['core_0014', 'core_0015']`

**Retrieved Top 5:** `['core_0014', 'core_0015', 'core_0040', 'core_0039', 'core_0041']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 3. Who should be offered PET-CT scanning before treatment for lung cancer?

**Ground truth:** `['core_0014', 'core_0015']`

**Retrieved Top 5:** `['core_0015', 'core_0032', 'core_0016', 'core_0014', 'core_0031']`

**Relevant flags:** `[True, False, False, True, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 4. Should MRI be routinely used to assess the T-stage of the primary tumour in NSCLC, and when is MRI specifically indicated instead?

**Ground truth:** `['core_0016', 'core_0017']`

**Retrieved Top 5:** `['core_0016', 'core_0017', 'core_0041', 'core_0038', 'core_0040']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 5. What is EBUS-TBNA used for in the diagnostic pathway for lung cancer, and why was clinical audit of EBUS-TBNA and EUS-FNA recommended by the committee?

**Ground truth:** `['core_0016', 'core_0017', 'core_0018']`

**Retrieved Top 5:** `['core_0018', 'core_0019', 'core_0022', 'core_0034', 'core_0020']`

**Relevant flags:** `[True, False, False, False, False]`

- Precision@5: `0.200`
- Recall@5: `0.333`
- Hit@5: `1`
- MRR@5: `1.000`

### 6. What guidance should be followed for next-generation sequencing (NGS) panel testing to guide lung cancer treatment, and what quality requirement applies to tissue samples taken for pathological diagnosis?

**Ground truth:** `['core_0018', 'core_0019']`

**Retrieved Top 5:** `['core_0019', 'core_0208', 'core_0018', 'core_0209', 'core_0180']`

**Relevant flags:** `[True, False, True, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 7. When choosing which lesion to biopsy in a patient with a peripheral primary tumour, should enlarged intrathoracic nodes or the primary lesion be prioritised, and under what size threshold are nodes considered enlarged?

**Ground truth:** `['core_0029']`

**Retrieved Top 5:** `['core_0029', 'core_0033', 'core_0034', 'core_0032', 'core_0031']`

**Relevant flags:** `[True, False, False, False, False]`

- Precision@5: `0.200`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 8. When is flexible bronchoscopy recommended for people with a central lung lesion on CT?

**Ground truth:** `['core_0030']`

**Retrieved Top 5:** `['core_0030', 'core_0014', 'core_0023', 'core_0022', 'core_0029']`

**Relevant flags:** `[True, False, False, False, False]`

- Precision@5: `0.200`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 9. What is the recommended staging pathway for intrathoracic lymph nodes in patients with a low probability of nodal malignancy versus those with enlarged (≥10 mm) intrathoracic lymph nodes who could have curative-intent treatment?

**Ground truth:** `['core_0031', 'core_0032', 'core_0033']`

**Retrieved Top 5:** `['core_0033', 'core_0031', 'core_0032', 'core_0034', 'core_0022']`

**Relevant flags:** `[True, True, True, False, False]`

- Precision@5: `0.600`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 10. When should surgical mediastinal staging be considered after EBUS-TBNA or EUS-FNA results?

**Ground truth:** `['core_0033', 'core_0034']`

**Retrieved Top 5:** `['core_0022', 'core_0034', 'core_0023', 'core_0033', 'core_0035']`

**Relevant flags:** `[False, True, False, True, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `0.500`

### 11. What brain imaging is recommended for people with clinical stage 1, stage 2, and stage 3 NSCLC who are being considered for curative-intent treatment?

**Ground truth:** `['core_0038', 'core_0039', 'core_0040', 'core_0041']`

**Retrieved Top 5:** `['core_0039', 'core_0038', 'core_0040', 'core_0043', 'core_0041']`

**Relevant flags:** `[True, True, True, False, True]`

- Precision@5: `0.800`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 12. What is the recommended first-line imaging test for a patient with localised signs or symptoms of bone metastasis, and what should be done if the result is negative or inconclusive?

**Ground truth:** `['core_0041', 'core_0042']`

**Retrieved Top 5:** `['core_0042', 'core_0041', 'core_0038', 'core_0043', 'core_0172']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 13. In what circumstances should bone scintigraphy be avoided in lung cancer staging?

**Ground truth:** `['core_0041', 'core_0042']`

**Retrieved Top 5:** `['core_0042', 'core_0041', 'core_0015', 'core_0014', 'core_0022']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 14. What cardiovascular assessment criteria determine whether a person with NSCLC can proceed to surgery without further cardiac investigation versus needing a cardiology review?

**Ground truth:** `['core_0058', 'core_0059']`

**Retrieved Top 5:** `['core_0058', 'core_0059', 'core_0057', 'core_0071', 'core_0070']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 15. What lung function tests should be performed before proceeding with curative-intent treatment for lung cancer, and what threshold is used for shuttle walk testing to assess fitness for surgery?

**Ground truth:** `['core_0062']`

**Retrieved Top 5:** `['core_0062', 'core_0070', 'core_0071', 'core_0063', 'core_0180']`

**Relevant flags:** `[True, False, False, False, False]`

- Precision@5: `0.200`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 16. What type of surgery should be offered to people with NSCLC who are well enough for curative-intent treatment, and when is more extensive surgery such as pneumonectomy indicated?

**Ground truth:** `['core_0065', 'core_0066']`

**Retrieved Top 5:** `['core_0066', 'core_0065', 'core_0084', 'core_0085', 'core_0067']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 17. What treatment options are recommended for people with stage 1 to 2a NSCLC who decline lobectomy or in whom it is contraindicated?

**Ground truth:** `['core_0069']`

**Retrieved Top 5:** `['core_0069', 'core_0072', 'core_0076', 'core_0071', 'core_0065']`

**Relevant flags:** `[True, False, False, False, False]`

- Precision@5: `0.200`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 18. What are the two recommended conventionally fractionated radical radiotherapy dose regimens for NSCLC?

**Ground truth:** `['core_0074', 'core_0075']`

**Retrieved Top 5:** `['core_0077', 'core_0078', 'core_0076', 'core_0075', 'core_0074']`

**Relevant flags:** `[False, False, False, True, True]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `0.250`

### 19. For people with operable stage 3a N2 NSCLC having chemoradiotherapy and surgery, how long after completing chemoradiotherapy should surgery be scheduled, and what benefits should be discussed with the person before starting this combined treatment?

**Ground truth:** `['core_0085', 'core_0086']`

**Retrieved Top 5:** `['core_0085', 'core_0086', 'core_0089', 'core_0090', 'core_0091']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 20. For which patients with resectable NSCLC is nivolumab in combination with chemotherapy recommended as neoadjuvant treatment?

**Ground truth:** `['core_0093']`

**Retrieved Top 5:** `['core_0093', 'core_0095', 'core_0094', 'core_0129', 'core_0090']`

**Relevant flags:** `[True, False, False, False, False]`

- Precision@5: `0.200`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 21. For which patients with resected NSCLC is osimertinib recommended as adjuvant treatment, and when should it be stopped?

**Ground truth:** `['core_0098', 'core_0099']`

**Retrieved Top 5:** `['core_0099', 'core_0098', 'core_0100', 'core_0101', 'core_0094']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 22. For which patients with resected NSCLC is alectinib recommended as adjuvant treatment?

**Ground truth:** `['core_0100', 'core_0101']`

**Retrieved Top 5:** `['core_0101', 'core_0100', 'core_0098', 'core_0099', 'core_0094']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 23. What first-line chemotherapy regimen and radiotherapy schedule are recommended for people with limited-stage SCLC?

**Ground truth:** `['core_0135', 'core_0136']`

**Retrieved Top 5:** `['core_0138', 'core_0154', 'core_0141', 'core_0139', 'core_0155']`

**Relevant flags:** `[False, False, False, False, False]`

- Precision@5: `0.000`
- Recall@5: `0.000`
- Hit@5: `0`
- MRR@5: `0.000`

### 24. What is the recommended dose and fractionation for prophylactic cranial irradiation in limited-stage SCLC, and under what condition should it be offered?

**Ground truth:** `['core_0138', 'core_0139']`

**Retrieved Top 5:** `['core_0139', 'core_0138', 'core_0189', 'core_0187', 'core_0151']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 25. What chemotherapy is recommended as first-line treatment for extensive-stage SCLC, and what is the maximum number of cycles that should be offered?

**Ground truth:** `['core_0143', 'core_0144']`

**Retrieved Top 5:** `['core_0145', 'core_0144', 'core_0154', 'core_0155', 'core_0157']`

**Relevant flags:** `[False, True, False, False, False]`

- Precision@5: `0.200`
- Recall@5: `0.500`
- Hit@5: `1`
- MRR@5: `0.500`

### 26. What treatment options are available for people with SCLC that has relapsed after first-line treatment, and what should patients be told if their disease did not respond to first-line treatment?

**Ground truth:** `['core_0154', 'core_0155', 'core_0156']`

**Retrieved Top 5:** `['core_0153', 'core_0152', 'core_0157', 'core_0156', 'core_0154']`

**Relevant flags:** `[False, False, False, True, True]`

- Precision@5: `0.400`
- Recall@5: `0.667`
- Hit@5: `1`
- MRR@5: `0.250`

### 27. What interventions should be offered to people with lung cancer who have impending endobronchial obstruction?

**Ground truth:** `['core_0160', 'core_0161']`

**Retrieved Top 5:** `['core_0161', 'core_0160', 'core_0165', 'core_0014', 'core_0056']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 28. What treatment options are available for people presenting with superior vena cava obstruction due to lung cancer?

**Ground truth:** `['core_0167', 'core_0168']`

**Retrieved Top 5:** `['core_0168', 'core_0167', 'core_0160', 'core_0161', 'core_0033']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 29. What is the recommended management of dexamethasone dosing in people with symptomatic brain metastases from lung cancer?

**Ground truth:** `['core_0169', 'core_0170']`

**Retrieved Top 5:** `['core_0170', 'core_0169', 'core_0188', 'core_0043', 'core_0187']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 30. When is denosumab recommended for preventing skeletal-related events in people with bone metastases from lung cancer?

**Ground truth:** `['core_0172', 'core_0173']`

**Retrieved Top 5:** `['core_0173', 'core_0172', 'core_0171', 'core_0180', 'core_0129']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 31. Within how many weeks of completing treatment should people with lung cancer be offered an initial follow-up appointment, and who can lead protocol-driven follow-up?

**Ground truth:** `['core_0175', 'core_0176']`

**Retrieved Top 5:** `['core_0176', 'core_0175', 'core_0177', 'core_0131', 'core_0053']`

**Relevant flags:** `[True, True, False, False, False]`

- Precision@5: `0.400`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

### 32. Should surgery for lung cancer be postponed to allow a person time to stop smoking first?

**Ground truth:** `['core_0056']`

**Retrieved Top 5:** `['core_0056', 'core_0055', 'core_0054', 'core_0088', 'core_0089']`

**Relevant flags:** `[True, False, False, False, False]`

- Precision@5: `0.200`
- Recall@5: `1.000`
- Hit@5: `1`
- MRR@5: `1.000`

Evaluation time: 6.1s
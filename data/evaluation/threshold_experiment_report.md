# Pulmo Guide — Threshold Experiment

## Retrieval Configuration

- Embedding: `BAAI/bge-small-en-v1.5`
- Semantic weight: `70%`
- BM25 weight: `30%`
- Top-K: `5`
- Reranker: `None`

## Threshold Comparison

| Threshold | Accepted | Rejected | Acceptance % | Correct | False | Missed | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |
| 0.25 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |
| 0.30 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |
| 0.35 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |
| 0.40 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |
| 0.45 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |
| 0.50 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |
| 0.55 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |
| 0.60 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |
| 0.65 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |
| 0.70 | 32 | 0 | 1.000 | 31 | 1 | 0 | 0.969 | 1.000 |

## Per-Question Top Scores

### 1. In what specific clinical scenario should sputum cytology be used for investigating suspected lung cancer?

- Top-1 hybrid score: `0.9697`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0013', 'core_0014']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0013` | 0.9697 | True |
| 2 | `core_0014` | 0.9226 | True |
| 3 | `core_0050` | 0.7863 | False |
| 4 | `core_0049` | 0.7301 | False |
| 5 | `core_0033` | 0.5967 | False |

### 2. What should be included in the contrast-enhanced chest CT scan offered to people with known or suspected lung cancer, and what precaution applies to people with renal impairment?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0014', 'core_0015']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0014` | 1.0000 | True |
| 2 | `core_0015` | 0.9033 | True |
| 3 | `core_0040` | 0.7353 | False |
| 4 | `core_0039` | 0.6958 | False |
| 5 | `core_0041` | 0.6644 | False |

### 3. Who should be offered PET-CT scanning before treatment for lung cancer?

- Top-1 hybrid score: `0.9930`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0014', 'core_0015']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0015` | 0.9930 | True |
| 2 | `core_0032` | 0.8458 | False |
| 3 | `core_0016` | 0.8269 | False |
| 4 | `core_0014` | 0.8244 | True |
| 5 | `core_0031` | 0.7591 | False |

### 4. Should MRI be routinely used to assess the T-stage of the primary tumour in NSCLC, and when is MRI specifically indicated instead?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0016', 'core_0017']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0016` | 1.0000 | True |
| 2 | `core_0017` | 0.9699 | True |
| 3 | `core_0041` | 0.6782 | False |
| 4 | `core_0038` | 0.6461 | False |
| 5 | `core_0040` | 0.6396 | False |

### 5. What is EBUS-TBNA used for in the diagnostic pathway for lung cancer, and why was clinical audit of EBUS-TBNA and EUS-FNA recommended by the committee?

- Top-1 hybrid score: `0.9568`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0016', 'core_0017', 'core_0018']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0018` | 0.9568 | True |
| 2 | `core_0019` | 0.9469 | False |
| 3 | `core_0022` | 0.8468 | False |
| 4 | `core_0034` | 0.8375 | False |
| 5 | `core_0020` | 0.8177 | False |

### 6. What guidance should be followed for next-generation sequencing (NGS) panel testing to guide lung cancer treatment, and what quality requirement applies to tissue samples taken for pathological diagnosis?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0018', 'core_0019']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0019` | 1.0000 | True |
| 2 | `core_0208` | 0.8077 | False |
| 3 | `core_0018` | 0.7957 | True |
| 4 | `core_0209` | 0.7685 | False |
| 5 | `core_0180` | 0.6835 | False |

### 7. When choosing which lesion to biopsy in a patient with a peripheral primary tumour, should enlarged intrathoracic nodes or the primary lesion be prioritised, and under what size threshold are nodes considered enlarged?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0029']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0029` | 1.0000 | True |
| 2 | `core_0033` | 0.7423 | False |
| 3 | `core_0034` | 0.6924 | False |
| 4 | `core_0032` | 0.6340 | False |
| 5 | `core_0031` | 0.6227 | False |

### 8. When is flexible bronchoscopy recommended for people with a central lung lesion on CT?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0030']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0030` | 1.0000 | True |
| 2 | `core_0014` | 0.5804 | False |
| 3 | `core_0023` | 0.5781 | False |
| 4 | `core_0022` | 0.5707 | False |
| 5 | `core_0029` | 0.5566 | False |

### 9. What is the recommended staging pathway for intrathoracic lymph nodes in patients with a low probability of nodal malignancy versus those with enlarged (≥10 mm) intrathoracic lymph nodes who could have curative-intent treatment?

- Top-1 hybrid score: `0.9715`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0031', 'core_0032', 'core_0033']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0033` | 0.9715 | True |
| 2 | `core_0031` | 0.9408 | True |
| 3 | `core_0032` | 0.8848 | True |
| 4 | `core_0034` | 0.8628 | False |
| 5 | `core_0022` | 0.7554 | False |

### 10. When should surgical mediastinal staging be considered after EBUS-TBNA or EUS-FNA results?

- Top-1 hybrid score: `0.9415`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0033', 'core_0034']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0022` | 0.9415 | False |
| 2 | `core_0034` | 0.9196 | True |
| 3 | `core_0023` | 0.8871 | False |
| 4 | `core_0033` | 0.8545 | True |
| 5 | `core_0035` | 0.8514 | False |

### 11. What brain imaging is recommended for people with clinical stage 1, stage 2, and stage 3 NSCLC who are being considered for curative-intent treatment?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0038', 'core_0039', 'core_0040', 'core_0041']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0039` | 1.0000 | True |
| 2 | `core_0038` | 0.9463 | True |
| 3 | `core_0040` | 0.9462 | True |
| 4 | `core_0043` | 0.9252 | False |
| 5 | `core_0041` | 0.8379 | True |

### 12. What is the recommended first-line imaging test for a patient with localised signs or symptoms of bone metastasis, and what should be done if the result is negative or inconclusive?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0041', 'core_0042']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0042` | 1.0000 | True |
| 2 | `core_0041` | 0.8266 | True |
| 3 | `core_0038` | 0.5795 | False |
| 4 | `core_0043` | 0.5681 | False |
| 5 | `core_0172` | 0.5585 | False |

### 13. In what circumstances should bone scintigraphy be avoided in lung cancer staging?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0041', 'core_0042']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0042` | 1.0000 | True |
| 2 | `core_0041` | 0.8472 | True |
| 3 | `core_0015` | 0.7912 | False |
| 4 | `core_0014` | 0.7412 | False |
| 5 | `core_0022` | 0.6848 | False |

### 14. What cardiovascular assessment criteria determine whether a person with NSCLC can proceed to surgery without further cardiac investigation versus needing a cardiology review?

- Top-1 hybrid score: `0.8354`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0058', 'core_0059']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0058` | 0.8354 | True |
| 2 | `core_0059` | 0.7993 | True |
| 3 | `core_0057` | 0.7952 | False |
| 4 | `core_0071` | 0.7110 | False |
| 5 | `core_0070` | 0.6664 | False |

### 15. What lung function tests should be performed before proceeding with curative-intent treatment for lung cancer, and what threshold is used for shuttle walk testing to assess fitness for surgery?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0062']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0062` | 1.0000 | True |
| 2 | `core_0070` | 0.7346 | False |
| 3 | `core_0071` | 0.6809 | False |
| 4 | `core_0063` | 0.6257 | False |
| 5 | `core_0180` | 0.5940 | False |

### 16. What type of surgery should be offered to people with NSCLC who are well enough for curative-intent treatment, and when is more extensive surgery such as pneumonectomy indicated?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0065', 'core_0066']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0066` | 1.0000 | True |
| 2 | `core_0065` | 0.9154 | True |
| 3 | `core_0084` | 0.7939 | False |
| 4 | `core_0085` | 0.7788 | False |
| 5 | `core_0067` | 0.7660 | False |

### 17. What treatment options are recommended for people with stage 1 to 2a NSCLC who decline lobectomy or in whom it is contraindicated?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0069']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0069` | 1.0000 | True |
| 2 | `core_0072` | 0.8675 | False |
| 3 | `core_0076` | 0.8516 | False |
| 4 | `core_0071` | 0.8002 | False |
| 5 | `core_0065` | 0.7737 | False |

### 18. What are the two recommended conventionally fractionated radical radiotherapy dose regimens for NSCLC?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0074', 'core_0075']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0077` | 1.0000 | False |
| 2 | `core_0078` | 0.8705 | False |
| 3 | `core_0076` | 0.7964 | False |
| 4 | `core_0075` | 0.7634 | True |
| 5 | `core_0074` | 0.7497 | True |

### 19. For people with operable stage 3a N2 NSCLC having chemoradiotherapy and surgery, how long after completing chemoradiotherapy should surgery be scheduled, and what benefits should be discussed with the person before starting this combined treatment?

- Top-1 hybrid score: `0.9992`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0085', 'core_0086']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0085` | 0.9992 | True |
| 2 | `core_0086` | 0.9893 | True |
| 3 | `core_0089` | 0.7876 | False |
| 4 | `core_0090` | 0.7519 | False |
| 5 | `core_0091` | 0.6999 | False |

### 20. For which patients with resectable NSCLC is nivolumab in combination with chemotherapy recommended as neoadjuvant treatment?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0093']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0093` | 1.0000 | True |
| 2 | `core_0095` | 0.8092 | False |
| 3 | `core_0094` | 0.7707 | False |
| 4 | `core_0129` | 0.6384 | False |
| 5 | `core_0090` | 0.5804 | False |

### 21. For which patients with resected NSCLC is osimertinib recommended as adjuvant treatment, and when should it be stopped?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0098', 'core_0099']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0099` | 1.0000 | True |
| 2 | `core_0098` | 0.9533 | True |
| 3 | `core_0100` | 0.8202 | False |
| 4 | `core_0101` | 0.7310 | False |
| 5 | `core_0094` | 0.6290 | False |

### 22. For which patients with resected NSCLC is alectinib recommended as adjuvant treatment?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0100', 'core_0101']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0101` | 1.0000 | True |
| 2 | `core_0100` | 0.9257 | True |
| 3 | `core_0098` | 0.7413 | False |
| 4 | `core_0099` | 0.7231 | False |
| 5 | `core_0094` | 0.6773 | False |

### 23. What first-line chemotherapy regimen and radiotherapy schedule are recommended for people with limited-stage SCLC?

- Top-1 hybrid score: `0.9852`
- Relevant chunk retrieved: `False`
- Ground truth: `['core_0135', 'core_0136']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0138` | 0.9852 | False |
| 2 | `core_0154` | 0.9661 | False |
| 3 | `core_0141` | 0.9650 | False |
| 4 | `core_0139` | 0.9195 | False |
| 5 | `core_0155` | 0.9137 | False |

### 24. What is the recommended dose and fractionation for prophylactic cranial irradiation in limited-stage SCLC, and under what condition should it be offered?

- Top-1 hybrid score: `0.9372`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0138', 'core_0139']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0139` | 0.9372 | True |
| 2 | `core_0138` | 0.9127 | True |
| 3 | `core_0189` | 0.8784 | False |
| 4 | `core_0187` | 0.8705 | False |
| 5 | `core_0151` | 0.8634 | False |

### 25. What chemotherapy is recommended as first-line treatment for extensive-stage SCLC, and what is the maximum number of cycles that should be offered?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0143', 'core_0144']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0145` | 1.0000 | False |
| 2 | `core_0144` | 0.9607 | True |
| 3 | `core_0154` | 0.8987 | False |
| 4 | `core_0155` | 0.8455 | False |
| 5 | `core_0157` | 0.8347 | False |

### 26. What treatment options are available for people with SCLC that has relapsed after first-line treatment, and what should patients be told if their disease did not respond to first-line treatment?

- Top-1 hybrid score: `0.9892`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0154', 'core_0155', 'core_0156']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0153` | 0.9892 | False |
| 2 | `core_0152` | 0.9886 | False |
| 3 | `core_0157` | 0.8654 | False |
| 4 | `core_0156` | 0.8623 | True |
| 5 | `core_0154` | 0.8540 | True |

### 27. What interventions should be offered to people with lung cancer who have impending endobronchial obstruction?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0160', 'core_0161']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0161` | 1.0000 | True |
| 2 | `core_0160` | 0.9360 | True |
| 3 | `core_0165` | 0.6888 | False |
| 4 | `core_0014` | 0.6751 | False |
| 5 | `core_0056` | 0.6738 | False |

### 28. What treatment options are available for people presenting with superior vena cava obstruction due to lung cancer?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0167', 'core_0168']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0168` | 1.0000 | True |
| 2 | `core_0167` | 0.8576 | True |
| 3 | `core_0160` | 0.6070 | False |
| 4 | `core_0161` | 0.6033 | False |
| 5 | `core_0033` | 0.5769 | False |

### 29. What is the recommended management of dexamethasone dosing in people with symptomatic brain metastases from lung cancer?

- Top-1 hybrid score: `0.9620`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0169', 'core_0170']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0170` | 0.9620 | True |
| 2 | `core_0169` | 0.9436 | True |
| 3 | `core_0188` | 0.6167 | False |
| 4 | `core_0043` | 0.5942 | False |
| 5 | `core_0187` | 0.5865 | False |

### 30. When is denosumab recommended for preventing skeletal-related events in people with bone metastases from lung cancer?

- Top-1 hybrid score: `0.9943`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0172', 'core_0173']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0173` | 0.9943 | True |
| 2 | `core_0172` | 0.9353 | True |
| 3 | `core_0171` | 0.5144 | False |
| 4 | `core_0180` | 0.4881 | False |
| 5 | `core_0129` | 0.4731 | False |

### 31. Within how many weeks of completing treatment should people with lung cancer be offered an initial follow-up appointment, and who can lead protocol-driven follow-up?

- Top-1 hybrid score: `0.9973`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0175', 'core_0176']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0176` | 0.9973 | True |
| 2 | `core_0175` | 0.9770 | True |
| 3 | `core_0177` | 0.7762 | False |
| 4 | `core_0131` | 0.6254 | False |
| 5 | `core_0053` | 0.5758 | False |

### 32. Should surgery for lung cancer be postponed to allow a person time to stop smoking first?

- Top-1 hybrid score: `1.0000`
- Relevant chunk retrieved: `True`
- Ground truth: `['core_0056']`

| Rank | Chunk ID | Score | Relevant |
|---:|---|---:|---|
| 1 | `core_0056` | 1.0000 | True |
| 2 | `core_0055` | 0.8302 | False |
| 3 | `core_0054` | 0.7835 | False |
| 4 | `core_0088` | 0.6270 | False |
| 5 | `core_0089` | 0.5926 | False |

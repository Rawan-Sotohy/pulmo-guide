# Pulmo Guide — Retrieval Latency Report

## Final Retrieval Configuration

- Embedding model: `BAAI/bge-small-en-v1.5`
- Retrieval method: `Hybrid Search`
- Semantic weight: `70%`
- BM25 weight: `30%`
- Alpha: `0.7`
- Final Top-K: `5`
- Reranking: `None`
- MS-MARCO: `None`
- NLI: `None`

## Latency Summary

| Metric | Value |
|---|---:|
| Number of queries | 32 |
| Average latency | 42.95 ms |
| Median latency | 40.35 ms |
| Minimum latency | 29.66 ms |
| Maximum latency | 78.29 ms |
| Standard deviation | 10.33 ms |

## Per-Question Latency

| # | Question | Latency (ms) |
|---:|---|---:|
| 1 | In what specific clinical scenario should sputum cytology be used for investigating suspected lung cancer? | 35.08 |
| 2 | What should be included in the contrast-enhanced chest CT scan offered to people with known or suspected lung cancer, and what precaution applies to people with renal impairment? | 37.72 |
| 3 | Who should be offered PET-CT scanning before treatment for lung cancer? | 29.66 |
| 4 | Should MRI be routinely used to assess the T-stage of the primary tumour in NSCLC, and when is MRI specifically indicated instead? | 37.14 |
| 5 | What is EBUS-TBNA used for in the diagnostic pathway for lung cancer, and why was clinical audit of EBUS-TBNA and EUS-FNA recommended by the committee? | 43.93 |
| 6 | What guidance should be followed for next-generation sequencing (NGS) panel testing to guide lung cancer treatment, and what quality requirement applies to tissue samples taken for pathological diagnosis? | 41.15 |
| 7 | When choosing which lesion to biopsy in a patient with a peripheral primary tumour, should enlarged intrathoracic nodes or the primary lesion be prioritised, and under what size threshold are nodes considered enlarged? | 41.62 |
| 8 | When is flexible bronchoscopy recommended for people with a central lung lesion on CT? | 38.00 |
| 9 | What is the recommended staging pathway for intrathoracic lymph nodes in patients with a low probability of nodal malignancy versus those with enlarged (≥10 mm) intrathoracic lymph nodes who could have curative-intent treatment? | 55.22 |
| 10 | When should surgical mediastinal staging be considered after EBUS-TBNA or EUS-FNA results? | 35.37 |
| 11 | What brain imaging is recommended for people with clinical stage 1, stage 2, and stage 3 NSCLC who are being considered for curative-intent treatment? | 35.46 |
| 12 | What is the recommended first-line imaging test for a patient with localised signs or symptoms of bone metastasis, and what should be done if the result is negative or inconclusive? | 43.28 |
| 13 | In what circumstances should bone scintigraphy be avoided in lung cancer staging? | 31.59 |
| 14 | What cardiovascular assessment criteria determine whether a person with NSCLC can proceed to surgery without further cardiac investigation versus needing a cardiology review? | 37.01 |
| 15 | What lung function tests should be performed before proceeding with curative-intent treatment for lung cancer, and what threshold is used for shuttle walk testing to assess fitness for surgery? | 39.61 |
| 16 | What type of surgery should be offered to people with NSCLC who are well enough for curative-intent treatment, and when is more extensive surgery such as pneumonectomy indicated? | 39.28 |
| 17 | What treatment options are recommended for people with stage 1 to 2a NSCLC who decline lobectomy or in whom it is contraindicated? | 34.01 |
| 18 | What are the two recommended conventionally fractionated radical radiotherapy dose regimens for NSCLC? | 50.67 |
| 19 | For people with operable stage 3a N2 NSCLC having chemoradiotherapy and surgery, how long after completing chemoradiotherapy should surgery be scheduled, and what benefits should be discussed with the person before starting this combined treatment? | 78.29 |
| 20 | For which patients with resectable NSCLC is nivolumab in combination with chemotherapy recommended as neoadjuvant treatment? | 65.34 |
| 21 | For which patients with resected NSCLC is osimertinib recommended as adjuvant treatment, and when should it be stopped? | 60.10 |
| 22 | For which patients with resected NSCLC is alectinib recommended as adjuvant treatment? | 46.56 |
| 23 | What first-line chemotherapy regimen and radiotherapy schedule are recommended for people with limited-stage SCLC? | 49.54 |
| 24 | What is the recommended dose and fractionation for prophylactic cranial irradiation in limited-stage SCLC, and under what condition should it be offered? | 54.17 |
| 25 | What chemotherapy is recommended as first-line treatment for extensive-stage SCLC, and what is the maximum number of cycles that should be offered? | 43.45 |
| 26 | What treatment options are available for people with SCLC that has relapsed after first-line treatment, and what should patients be told if their disease did not respond to first-line treatment? | 43.23 |
| 27 | What interventions should be offered to people with lung cancer who have impending endobronchial obstruction? | 33.58 |
| 28 | What treatment options are available for people presenting with superior vena cava obstruction due to lung cancer? | 32.20 |
| 29 | What is the recommended management of dexamethasone dosing in people with symptomatic brain metastases from lung cancer? | 38.34 |
| 30 | When is denosumab recommended for preventing skeletal-related events in people with bone metastases from lung cancer? | 37.32 |
| 31 | Within how many weeks of completing treatment should people with lung cancer be offered an initial follow-up appointment, and who can lead protocol-driven follow-up? | 41.09 |
| 32 | Should surgery for lung cancer be postponed to allow a person time to stop smoking first? | 45.32 |

# Cyber Threat Classification System

A practical Python project for classifying cyber threat incident text into common attack categories such as phishing, ransomware, malware, DDoS, insider threats, credential stuffing, data exfiltration, and zero-day exploits.

## Features

- Training pipeline using a synthetic cyber-threat dataset
- TF-IDF + logistic regression classifier
- Class-balanced training tuned against repeated stratified cross-validation
- Polished Streamlit dashboard with live classification as incident text changes
- Full category probability breakdowns and timestamped JSON reports
- CSV upload and bulk classification with detailed CSV and JSON exports
- Dataset insights and category distribution charts
- One-click model retraining workflow
- Automated tests for prediction quality

## Project structure

- `app.py` – Streamlit web interface
- `train_model.py` – trains and saves the model
- `cyber_threat_classifier.py` – reusable classifier implementation
- `data/threat_dataset.csv` – labeled sample incident dataset
- `tests/test_pipeline.py` – validation tests

## Setup

```bash
cd cyber-threat-classification-system
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Include `app.py`, `requirements.txt`,
   `cyber_threat_classifier.py`, and `threat_dataset.csv` (at the repository root)
   or `data/threat_dataset.csv`. The CSV is the training and insights dataset used
   by the app. The app also supports the CSV at the repository root.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), select **Create app**,
   choose the repository and branch, and set the main file path to `app.py`.
3. Deploy. The app reads the bundled dataset relative to its own location, so the
   dataset remains available regardless of the cloud server's working directory.

The bundled training CSV must have `text` and `label` columns. In the **Bulk report**
tab, uploaded CSVs need a `text` column; an optional `label` column is included in
the results as the actual label for comparison. The app trains a model on first
startup when no saved model is present, so committing the model file is optional.

## Retrain the model

```bash
python train_model.py
```

## Example classifications

- "Fake password reset email asks employees to confirm credentials" → Phishing
- "Ransomware encrypted production files and demanded bitcoin" → Ransomware
- "Botnet traffic overloaded public API endpoints for six hours" → DDoS
- "Privileged user copied customer data to an external storage drive" → Insider Threat

## Notes

Live analysis refers to classifying text as it is entered; the dashboard does not ingest network traffic or provide live infrastructure monitoring. This project contains only 37 synthetic examples across 9 categories, so measured accuracy is limited and can vary with the evaluation split. The current model configuration averaged about 67% accuracy in a repeated 3-fold cross-validation experiment on this dataset; that is a small-data benchmark, not a guarantee of production performance. Model confidence values are estimates, not verified incident severity. For production use, collect more diverse, representative labeled incidents, evaluate on a separate holdout set, and have a security analyst review findings.

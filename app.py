from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from cyber_threat_classifier import ThreatClassifier, resolve_dataset_path


ROOT = Path(__file__).resolve().parent
DATASET_PATH = resolve_dataset_path(ROOT)
MODEL_DIR = ROOT / "model"

st.set_page_config(
    page_title="ThreatLens | Cyber Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #e7eefb;
        --muted: #91a2bc;
        --panel: #111c2c;
        --line: rgba(148, 163, 184, .16);
        --mint: #58e0bd;
        --blue: #70a7ff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 75% -10%, rgba(52, 112, 190, .18), transparent 38%),
            linear-gradient(160deg, #08111e 0%, #0b1422 52%, #0a1220 100%);
        color: var(--ink);
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; letter-spacing: -.03em; }
    h1 { font-size: clamp(2.1rem, 4vw, 3.2rem) !important; }
    [data-testid="stSidebar"] { background: #0c1726; border-right: 1px solid var(--line); }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(20, 34, 52, .96), rgba(15, 27, 43, .96));
        border: 1px solid var(--line);
        padding: 18px 20px;
        border-radius: 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Segoe UI', sans-serif; }
    [data-testid="stTabs"] button { font-weight: 600; }
    [data-testid="stTextArea"] textarea {
        background: #0e1a2a;
        border: 1px solid rgba(148, 163, 184, .24);
        border-radius: 14px;
        color: var(--ink);
        line-height: 1.65;
    }
    [data-testid="stFileUploader"] section {
        background: rgba(17, 28, 44, .7);
        border: 1px dashed rgba(112, 167, 255, .45);
        border-radius: 16px;
    }
    .hero {
        padding: 1.3rem 0 1.6rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.5rem;
    }
    .eyebrow {
        color: var(--mint);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero-copy { color: var(--muted); font-size: 1.05rem; max-width: 720px; }
    .live-badge {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        color: #a7f3dc;
        background: rgba(32, 176, 139, .12);
        border: 1px solid rgba(88, 224, 189, .28);
        border-radius: 999px;
        padding: 6px 11px;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .06em;
    }
    .live-dot {
        width: 7px;
        height: 7px;
        background: var(--mint);
        border-radius: 50%;
        box-shadow: 0 0 12px var(--mint);
    }
    .report-card {
        background: linear-gradient(145deg, rgba(20, 34, 52, .96), rgba(13, 25, 40, .96));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        margin: .5rem 0 1rem;
    }
    .report-label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .1em; }
    .report-value { color: var(--ink); font: 700 1.55rem 'Segoe UI', sans-serif; margin-top: .25rem; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(88, 224, 189, .35);
        background: linear-gradient(110deg, #187e70, #176b84);
        color: #fff;
        font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--mint);
        color: white;
    }
    .footnote { color: var(--muted); font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_classifier() -> ThreatClassifier:
    if not DATASET_PATH.is_file():
        st.error(f"Training dataset not found: `{DATASET_PATH}`")
        st.info(
            "Add `threat_dataset.csv` to the GitHub repository root or place it "
            "at `data/threat_dataset.csv`, commit and push the change, then reboot the app. "
            "The CSV must contain `text` and `label` columns."
        )
        st.stop()

    classifier = ThreatClassifier(MODEL_DIR)
    if not classifier.model_path.exists():
        classifier.train_from_csv(DATASET_PATH)
    else:
        classifier.load()
        if classifier.model_version != classifier.MODEL_VERSION:
            classifier.train_from_csv(DATASET_PATH)
    return classifier


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'text' column.")

    results = df.copy()
    texts = results["text"].fillna("").astype(str).tolist()
    distributions = load_classifier().predict_distribution_batch(texts)

    results["predicted_label"] = [scores[0][0] for scores in distributions]
    results["confidence"] = [scores[0][1] for scores in distributions]
    results["confidence_percent"] = results["confidence"] * 100
    results["probability_distribution"] = [
        json.dumps(
            [
                {"label": label, "probability": round(probability, 6)}
                for label, probability in scores
            ],
            ensure_ascii=False,
        )
        for scores in distributions
    ]
    results["analysis_timestamp_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    if "label" in results.columns:
        results["actual_label"] = results["label"]
    else:
        results["actual_label"] = ""

    return results[
        [
            "text",
            "actual_label",
            "predicted_label",
            "confidence",
            "confidence_percent",
            "probability_distribution",
            "analysis_timestamp_utc",
        ]
    ]


def apply_selected_sample() -> None:
    st.session_state["incident_text"] = sample_incidents[
        st.session_state["sample_incident"]
    ]


def to_report_json(report: dict) -> bytes:
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")


sample_incidents = {
    "Phishing": "A fake password reset email asked employees to confirm credentials through a cloned login page.",
    "Ransomware": "Ransomware encrypted the production file share and left a payment notice demanding bitcoin.",
    "DDoS": "A distributed denial of service attack flooded the public API with millions of requests.",
    "Insider Threat": "A contractor downloaded restricted customer records and shared them with a personal email account.",
    "Credential Stuffing": "Attackers used leaked password lists against the SSO portal in a credential stuffing campaign.",
    "Zero-Day Exploit": "A newly disclosed browser vulnerability was exploited before the security patch was deployed.",
    "Data Exfiltration": "Sensitive financial records were transferred to an external archive over an encrypted SSH tunnel.",
}

classifier = load_classifier()

with st.sidebar:
    st.markdown("## 🛡️ ThreatLens")
    st.caption("AI-assisted incident triage")
    st.divider()
    st.markdown("**CLASSIFIER STATUS**")
    st.markdown(
        '<span class="live-badge"><span class="live-dot"></span> MODEL READY</span>',
        unsafe_allow_html=True,
    )
    st.markdown("### Supported categories")
    for label in classifier.available_labels():
        st.caption(f"• {label}")

    st.divider()
    if st.button("Retrain model", width="stretch"):
        with st.spinner("Training the classifier..."):
            classifier.train_from_csv(DATASET_PATH)
        st.success("Model retrained successfully.")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Cyber threat intelligence · incident triage</div>
      <h1>See the signal in every incident.</h1>
      <div class="hero-copy">
        Classify incident descriptions as you type, inspect every category score,
        and export a timestamped report for your records.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

classification_tab, bulk_upload_tab, insights_tab = st.tabs(
    ["Live analysis", "Bulk report", "Dataset insights"]
)

with classification_tab:
    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        st.markdown("### Analyze an incident")
        st.markdown(
            '<span class="live-badge"><span class="live-dot"></span> LIVE · UPDATES AS YOU TYPE</span>',
            unsafe_allow_html=True,
        )
        st.caption("Choose an example or enter an incident description.")

        default_sample = next(iter(sample_incidents))
        st.session_state.setdefault("sample_incident", default_sample)
        st.session_state.setdefault("incident_text", sample_incidents[default_sample])
        st.selectbox(
            "Load a sample incident",
            list(sample_incidents),
            key="sample_incident",
            on_change=apply_selected_sample,
        )
        user_text = st.text_area(
            "Incident description",
            key="incident_text",
            height=205,
            placeholder="Describe the observed incident, affected systems, and relevant activity...",
            label_visibility="collapsed",
        )
        st.caption("Analysis is performed locally with this project's trained model.")

    with right:
        st.markdown("### Analysis report")
        if not user_text.strip():
            st.info("Enter an incident description to see a live classification report.")
        else:
            with st.spinner("Analyzing incident..."):
                distribution = classifier.predict_distribution(user_text)

            predicted_label, confidence = distribution[0]
            generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            confidence_percent = confidence * 100
            report = {
                "report_generated_at_utc": generated_at,
                "model": "TF-IDF + Logistic Regression",
                "incident_description": user_text,
                "predicted_category": predicted_label,
                "confidence": round(confidence, 6),
                "confidence_percent": round(confidence_percent, 2),
                "category_probabilities": [
                    {
                        "category": label,
                        "probability": round(probability, 6),
                        "probability_percent": round(probability * 100, 2),
                    }
                    for label, probability in distribution
                ],
                "limitations": (
                    "Model probabilities are estimates from a synthetic training "
                    "dataset; validate findings with a security analyst."
                ),
            }

            st.markdown(
                f"""
                <div class="report-card">
                  <div class="report-label">Most likely category</div>
                  <div class="report-value">{predicted_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            metric_one, metric_two = st.columns(2)
            metric_one.metric("Model confidence", f"{confidence_percent:.1f}%")
            metric_two.metric("Categories evaluated", len(distribution))
            st.caption(f"Report generated: {generated_at} · UTC")

            st.markdown("#### Category probability breakdown")
            probability_frame = pd.DataFrame(
                distribution, columns=["Category", "Probability"]
            )
            st.bar_chart(
                probability_frame.set_index("Category")["Probability"],
                horizontal=True,
                color="#58e0bd",
            )
            with st.expander("View all category scores and report details"):
                st.dataframe(
                    probability_frame.assign(
                        Confidence=probability_frame["Probability"].map(
                            lambda score: f"{score:.1%}"
                        )
                    )[["Category", "Confidence"]],
                    width="stretch",
                    hide_index=True,
                )
                st.caption(report["limitations"])
                st.download_button(
                    "Download full JSON report",
                    data=to_report_json(report),
                    file_name="cyber_threat_report.json",
                    mime="application/json",
                    width="stretch",
                )

    st.markdown(
        '<p class="footnote">This dashboard classifies text; it does not monitor network traffic or replace incident response analysis.</p>',
        unsafe_allow_html=True,
    )

with bulk_upload_tab:
    st.markdown("### Analyze a batch")
    st.caption(
        "Upload a CSV with a `text` column. Each row receives a prediction, "
        "all category probabilities, and a UTC analysis timestamp."
    )
    uploaded_file = st.file_uploader(
        "Upload incident records", type=["csv"], help="Required column: text"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            classified = classify_dataframe(df)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Unable to process CSV: {exc}")
        else:
            st.success(f"Processed {len(classified)} incident records.")
            if not classified.empty:
                count_col, category_col, confidence_col = st.columns(3)
                count_col.metric("Records analyzed", len(classified))
                category_col.metric(
                    "Categories found", classified["predicted_label"].nunique()
                )
                confidence_col.metric(
                    "Average confidence",
                    f"{classified['confidence_percent'].mean():.1f}%",
                )
            st.dataframe(classified, width="stretch", hide_index=True)

            report = {
                "report_generated_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "model": "TF-IDF + Logistic Regression",
                "records_analyzed": len(classified),
                "limitations": (
                    "Model probabilities are estimates from a synthetic training "
                    "dataset; validate findings with a security analyst."
                ),
                "results": classified.to_dict(orient="records"),
            }
            csv_data = classified.to_csv(index=False).encode("utf-8")
            json_data = to_report_json(report)
            csv_col, json_col = st.columns(2)
            csv_col.download_button(
                "Download detailed CSV",
                data=csv_data,
                file_name="cyber_threat_report.csv",
                mime="text/csv",
                width="stretch",
            )
            json_col.download_button(
                "Download full JSON report",
                data=json_data,
                file_name="cyber_threat_report.json",
                mime="application/json",
                width="stretch",
            )

with insights_tab:
    st.markdown("### Training data overview")
    st.caption("Understand the category coverage behind the current classifier.")
    dataset = pd.read_csv(DATASET_PATH)
    summary = dataset["label"].value_counts().rename_axis("label").reset_index(
        name="count"
    )
    summary = summary.sort_values("count", ascending=False)

    total_records = len(dataset)
    total_categories = dataset["label"].nunique()
    insight_one, insight_two = st.columns(2)
    insight_one.metric("Training examples", f"{total_records:,}")
    insight_two.metric("Threat categories", total_categories)
    chart_col, table_col = st.columns([1.2, 0.8], gap="large")
    with chart_col:
        st.bar_chart(summary.set_index("label")["count"], color="#70a7ff")
    with table_col:
        st.dataframe(summary, width="stretch", hide_index=True)
    st.caption(
        "The included dataset is synthetic and intended for demonstration. "
        "Production use requires validation on representative incident data."
    )

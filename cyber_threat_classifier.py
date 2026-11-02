from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def resolve_dataset_path(base_dir: str | Path) -> Path:
    root = Path(base_dir)
    candidates = (
        root / "data" / "threat_dataset.csv",
        root / "threat_dataset.csv",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


class ThreatClassifier:
    MODEL_VERSION = 2

    def __init__(self, model_dir: str | Path | None = None):
        base_dir = Path(__file__).resolve().parent
        self.model_dir = Path(model_dir) if model_dir else base_dir / "model"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "threat_classifier.joblib"
        self.model = None
        self.vectorizer = None
        self.labels = []
        self.model_version = 0

    def train_from_csv(self, csv_path: str | Path) -> LogisticRegression:
        dataset = Path(csv_path)
        if not dataset.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        df = pd.read_csv(dataset)
        required_columns = {"text", "label"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

        texts = df["text"].fillna("").astype(str).tolist()
        labels = df["label"].fillna("").astype(str).tolist()

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            stop_words="english",
            sublinear_tf=True,
        )
        features = vectorizer.fit_transform(texts)

        classifier = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=3000,
        )
        classifier.fit(features, labels)

        self.model = classifier
        self.vectorizer = vectorizer
        self.labels = sorted(set(labels))
        self.model_version = self.MODEL_VERSION

        payload = {
            "model": classifier,
            "vectorizer": vectorizer,
            "labels": self.labels,
            "model_version": self.model_version,
        }
        joblib.dump(payload, self.model_path)
        return classifier

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"No trained model found at {self.model_path}")

        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.vectorizer = payload["vectorizer"]
        self.labels = payload.get("labels", [])
        self.model_version = payload.get("model_version", 0)
        return self

    def predict_distribution(self, text: str) -> list[tuple[str, float]]:
        return self.predict_distribution_batch([text])[0]

    def predict_distribution_batch(
        self, texts: list[str]
    ) -> list[list[tuple[str, float]]]:
        if self.model is None or self.vectorizer is None:
            self.load()

        if not texts:
            return []

        transformed = self.vectorizer.transform(texts)
        distributions = []
        for row in self.model.predict_proba(transformed):
            scores = [
                (str(label), float(probability))
                for label, probability in zip(self.model.classes_, row)
            ]
            distributions.append(
                sorted(scores, key=lambda item: item[1], reverse=True)
            )
        return distributions

    def predict_batch(self, texts: list[str]) -> list[Tuple[str, float]]:
        distributions = self.predict_distribution_batch(texts)
        return [distribution[0] for distribution in distributions]

    def predict(self, text: str) -> Tuple[str, float]:
        distribution = self.predict_distribution(text)
        prediction, confidence = distribution[0]
        return prediction, confidence

    def available_labels(self) -> list[str]:
        if not self.labels:
            if self.model is not None:
                return [str(label) for label in self.model.classes_]
            return []
        return list(self.labels)

    def dataset_summary(self, csv_path: str | Path):
        dataset = Path(csv_path)
        if not dataset.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        df = pd.read_csv(dataset)
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        counts = counts.sort_values("label")
        return counts

    def ensure_trained(self, dataset_path: str | Path):
        if self.model_path.exists():
            self.load()
            return self
        self.train_from_csv(dataset_path)
        return self


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    classifier = ThreatClassifier(root / "model")
    classifier.train_from_csv(resolve_dataset_path(root))
    print("Model trained and saved to", classifier.model_path)

from pathlib import Path

from cyber_threat_classifier import ThreatClassifier, resolve_dataset_path


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset_path = resolve_dataset_path(root)
    model_dir = root / "model"

    classifier = ThreatClassifier(model_dir)
    classifier.train_from_csv(dataset_path)
    print(f"Training complete. Saved model to {classifier.model_path}")
    print(f"Classes: {classifier.available_labels()}")


if __name__ == "__main__":
    main()

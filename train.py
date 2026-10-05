"""Train once, evaluate on held-out samples, and export the entire pipeline."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES

BASE_DIR = Path(__file__).resolve().parent


def train():
    dataset = load_breast_cancer(as_frame=True)
    class_mapping = dict(enumerate(dataset.target_names.tolist()))
    assert class_mapping == {0: "malignant", 1: "benign"}
    X = dataset.data[list(FEATURES.values())]
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, average="binary", pos_label=0, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "positive_class": "malignant", "positive_class_code": 0,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_class_order": ["malignant", "benign"],
    }
    joblib.dump(model, BASE_DIR / "model.pkl")
    reloaded = joblib.load(BASE_DIR / "model.pkl")
    assert (reloaded.predict(X_test) == predicted).all()
    sample = X_test.iloc[0]
    example = {field: float(sample[column]) for field, column in FEATURES.items()}
    metadata = {
        "dataset": "Wisconsin Breast Cancer Diagnostic Dataset",
        "samples": len(X), "original_feature_count": dataset.data.shape[1],
        "model_type": "StandardScaler + LogisticRegression (sklearn Pipeline)",
        "selected_features": list(FEATURES.values()),
        "api_feature_names": list(FEATURES), "feature_order": list(FEATURES.values()),
        "feature_mapping": FEATURES, "target_names": dataset.target_names.tolist(),
        "class_mapping": class_mapping, "random_seed": 42, "test_size": 0.2,
        "train_samples": len(X_train), "test_samples": len(X_test),
        "metrics": metrics, "sklearn_version": sklearn.__version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "example_dataset_row": int(sample.name),
        "example_true_class": class_mapping[int(y_test.loc[sample.name])],
    }
    for name, value in [("model_metadata.json", metadata), ("example_request.json", example)]:
        (BASE_DIR / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("Saved and reloaded pipeline; predictions match.")


if __name__ == "__main__":
    train()

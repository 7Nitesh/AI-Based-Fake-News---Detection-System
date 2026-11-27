"""Train and evaluate a TF-IDF + Logistic Regression text classifier.

Expected CSV columns: text,label
Optional: title (combined with text). Labels should be REAL or FAKE.
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_news.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fake_news_pipeline.joblib"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    if "title" in data.columns:
        data["content"] = data["title"].fillna("").astype(str) + " " + data["text"].fillna("").astype(str)
    else:
        data["content"] = data["text"].fillna("").astype(str)

    data = data[["content", "label"]].dropna()
    data["label"] = data["label"].astype(str).str.strip().str.upper()
    data = data[data["label"].isin(["REAL", "FAKE"])]
    data = data[data["content"].str.strip().str.len() > 0]

    counts = data["label"].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Dataset needs at least two REAL and two FAKE examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["content"], data["label"], test_size=0.25,
        random_state=42, stratify=data["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels: FAKE, REAL):")
    print(confusion_matrix(y_test, predictions, labels=["FAKE", "REAL"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()

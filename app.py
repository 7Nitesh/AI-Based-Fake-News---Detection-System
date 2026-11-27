"""Flask interface for the AI-Based Fake News Detection System."""
from pathlib import Path
import joblib
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_news_pipeline.joblib"

app = Flask(__name__)

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    news_text = ""
    error = None

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        if len(news_text) < 30:
            error = "Please enter at least 30 characters of news text."
        else:
            model = load_model()
            if model is None:
                error = "The model is not trained yet. Run: python train_model.py"
            else:
                label = str(model.predict([news_text])[0]).upper()
                prediction = "FAKE" if label in {"FAKE", "1", "FALSE"} else "REAL"
    return render_template("index.html", prediction=prediction, news_text=news_text, error=error)

if __name__ == "__main__":
    # Debug is disabled by default for safer local execution.
    app.run(host="127.0.0.1", port=5000, debug=False)

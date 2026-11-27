# AI-Based Fake News Detection System

**MCA Project**  
**Student:** Nitesh Panduranga Naik  
**USN:** SCA25MCA041

A Python machine-learning project that classifies submitted news text as **REAL** or **FAKE** using a TF-IDF text representation and Logistic Regression classifier. A Flask web interface allows a user to paste news content and view the model's prediction.

## Project structure

```
.
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── sample_news.csv
├── model/                 # generated after training; not committed
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Requirements

- Python 3.10 or newer
- pip

## Run locally

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Install packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Train the model:

   ```bash
   python train_model.py
   ```

5. Start the web application:

   ```bash
   python app.py
   ```

6. Open http://127.0.0.1:5000 in your browser.

## Dataset format

Use a CSV file at `data/sample_news.csv` with at least these columns:

| Column | Meaning |
|---|---|
| `text` | Full news article text |
| `label` | `REAL` or `FAKE` |
| `title` | Optional article title |

The included CSV is a **tiny synthetic demonstration dataset** so that the code can be tested. It is not a research-quality dataset and cannot support meaningful accuracy claims. For academic evaluation, replace it with a properly sourced, sufficiently large, labelled dataset, document its source and licence, clean duplicates, and evaluate on held-out data.

## Methodology

1. Load and validate labelled news text from CSV.
2. Combine title and article text when a title column is present.
3. Convert words and word pairs into TF-IDF numerical features.
4. Train a Logistic Regression classifier.
5. Print test-set accuracy, classification report and confusion matrix.
6. Save the trained pipeline locally and load it in the Flask app.

## Important limitation

This system detects patterns learned from its training data. It does **not** independently verify facts, check live sources, or guarantee that an article is true or false. A prediction must not be treated as proof. Dataset quality, class balance, source bias and changing news topics can strongly affect results.

## Student details

- **Name:** Nitesh Panduranga Naik
- **USN:** SCA25MCA041

## References

- Flask documentation: https://flask.palletsprojects.com/en/stable/quickstart/
- scikit-learn TF-IDF documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- scikit-learn Logistic Regression documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

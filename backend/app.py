'''Flask backend for Intelligent Fake News Detection

This API loads the previously saved Logistic Regression model and its matching TF‑IDF vectorizer
(and does NOT retrain anything). It exposes a single endpoint:
    POST /predict
which expects JSON of the form {"news_text": "..."} and returns a prediction.

The model and vectorizer files are automatically discovered from the ``models`` folder by
looking for the most‑recent files whose names start with
``final_corrected_logistic_regression`` and ``final_corrected_tfidf_vectorizer``.
''' 

import os
import json
import pickle
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score  # placeholder import to ensure scikit‑learn is available

# ---------------------------
# Configuration / paths
# ---------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def _latest_file(prefix: str) -> str:
    """Return the full path of the most‑recent file in ``MODELS_DIR`` that starts with *prefix*.
    Raises ``FileNotFoundError`` if no matching file exists.
    """
    candidates = [f for f in os.listdir(MODELS_DIR) if f.startswith(prefix) and f.endswith('.pkl')]
    if not candidates:
        raise FileNotFoundError(f"No model files with prefix '{prefix}' found in {MODELS_DIR}")
    # Sort by modification time, newest first
    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(MODELS_DIR, f)), reverse=True)
    return os.path.join(MODELS_DIR, candidates[0])

# Discover the latest corrected model / vectorizer
MODEL_PATH = _latest_file('final_corrected_logistic_regression')
VECTORIZER_PATH = _latest_file('final_corrected_tfidf_vectorizer')

# ---------------------------
# Load artifacts (read‑only)
# ---------------------------
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load Logistic Regression model from {MODEL_PATH}: {e}")

try:
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load TF‑IDF vectorizer from {VECTORIZER_PATH}: {e}")

# ---------------------------
# Flask app definition
# ---------------------------
app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, 'frontend', 'static'), static_url_path='/static')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether a news article is REAL (0) or FAKE (1).
    Expected JSON payload: {"news_text": "<article text>"}
    Returns JSON with ``prediction``, ``label`` and ``confidence`` (probability of the predicted class).
    """
    # Ensure request is JSON
    if not request.is_json:
        return jsonify({"error": "Invalid content type – expecting application/json"}), 400
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Malformed JSON"}), 400
    # Validate presence of ``news_text``
    news_text = data.get('news_text')
    if news_text is None:
        return jsonify({"error": "Missing 'news_text' field in JSON payload"}), 400
    if not isinstance(news_text, str) or not news_text.strip():
        return jsonify({"error": "'news_text' must be a non‑empty string"}), 400

    # Transform using the pre‑trained TF‑IDF vectorizer (no extra preprocessing)
    try:
        X = vectorizer.transform([news_text])
    except Exception as e:
        return jsonify({"error": f"Vectorizer transform failed: {e}"}), 500

    # Predict label and probability (if the model supports ``predict_proba``)
    try:
        label = int(model.predict(X)[0])
        # ``predict_proba`` returns probabilities for each class in order [0, 1]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][label]
        else:
            prob = None
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    prediction_str = "FAKE" if label == 1 else "REAL"
    response = {
        "prediction": prediction_str,
        "label": label,
        "confidence": round(float(prob), 4) if prob is not None else None
    }
    return jsonify(response)

# ---------------------------
# Serve the frontend index page
from flask import send_from_directory

FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')

@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')
# ---------------------------
# Run the server (development mode)
# ---------------------------
if __name__ == '__main__':
    # By default Flask runs on http://127.0.0.1:5000
    # Use ``debug=True`` for auto‑reload during development.
    app.run(host='0.0.0.0', port=5000, debug=True)

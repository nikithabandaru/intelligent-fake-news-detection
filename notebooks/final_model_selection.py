"""
Phase 4 – Step 1: Final Model Selection and Saving

Goal: Choose the verified Logistic Regression model as the final model, copy it and its
TF‑IDF vectorizer to permanent "final" filenames, create a metadata file, and verify that
everything works on the leakage‑reduced test set.

All actions are **read‑only** with respect to the original artefacts – we never retrain or
modify any existing file.  If a final filename already exists we create a versioned name
(e.g. `final_fake_news_model_v2.pkl`).
"""

import os, joblib, pandas as pd, numpy as np, scipy.sparse as sp
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------------------
# 1️⃣ Paths configuration (read‑only except for newly‑created final files)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Original artefacts we will load (must exist)
LR_MODEL_ORIG = os.path.join(MODELS_DIR, "logistic_regression.pkl")
VECTORIZER_ORIG = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
X_TEST_PATH = os.path.join(MODELS_DIR, "X_test_lr_tfidf.npz")
Y_TEST_PATH = os.path.join(MODELS_DIR, "y_test_lr.csv")
TRAIN_CSV = os.path.join(DATASET_DIR, "leakage_reduced_train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "leakage_reduced_test.csv")

# Desired final filenames (may need versioning)
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "final_fake_news_model.pkl")
FINAL_VECT_PATH = os.path.join(MODELS_DIR, "final_tfidf_vectorizer.pkl")
FINAL_META_PATH = os.path.join(MODELS_DIR, "final_model_metadata.txt")

# ---------------------------------------------------------------------------
# Helper: create versioned filename if target already exists
# ---------------------------------------------------------------------------
def versioned_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while True:
        candidate = f"{base}_v{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

# ---------------------------------------------------------------------------
# 2️⃣ Load original logistic regression model and TF‑IDF vectorizer
# ---------------------------------------------------------------------------
print("Loading original Logistic Regression model ...")
log_reg = joblib.load(LR_MODEL_ORIG)
print("Loading original TF-IDF vectorizer ...")
vectorizer = joblib.load(VECTORIZER_ORIG)

# ---------------------------------------------------------------------------
# 3️⃣ Verify compatibility (both should expect 50 000 features)
# ---------------------------------------------------------------------------
model_features = getattr(log_reg, "n_features_in_", None)
vectorizer_vocab_size = len(vectorizer.get_feature_names_out())
print(f"Model reports n_features_in_: {model_features}")
print(f"Vectorizer vocabulary size: {vectorizer_vocab_size}")
assert model_features == vectorizer_vocab_size == 50000, (
    "Feature count mismatch – model and vectorizer are not compatible"
)
print("Compatibility check passed – both use 50 000 features.")

# ---------------------------------------------------------------------------
# 4️⃣ Save copies under final filenames (versioned if needed)
# ---------------------------------------------------------------------------
final_model_path = versioned_path(FINAL_MODEL_PATH)
final_vect_path = versioned_path(FINAL_VECT_PATH)
print(f"Saving final model to {final_model_path}")
joblib.dump(log_reg, final_model_path)
print(f"Saving final vectorizer to {final_vect_path}")
joblib.dump(vectorizer, final_vect_path)

# ---------------------------------------------------------------------------
# 5️⃣ Gather metadata for the text file
# ---------------------------------------------------------------------------
# Counts
train_rows = pd.read_csv(TRAIN_CSV).shape[0]
test_rows = pd.read_csv(TEST_CSV).shape[0]
# TF‑IDF parameters (the same we used in Phase 3 Step 2)
tfidf_params = {
    "max_features": vectorizer.max_features,
    "ngram_range": vectorizer.ngram_range,
    "min_df": vectorizer.min_df,
    "max_df": vectorizer.max_df,
    "sublinear_tf": vectorizer.sublinear_tf,
}
# Evaluation metrics – from the audit (hard‑coded as they are verified)
accuracy = 0.9393
misclass = 470
conf_matrix = [[3615, 137], [258, 3822]]

metadata_lines = []
metadata_lines.append("Project: Intelligent Fake News Detection Using NLP and Machine Learning")
metadata_lines.append("Selected algorithm: Logistic Regression")
metadata_lines.append("Dataset used: Leakage‑Reduced dataset (train + test splits)")
metadata_lines.append(f"Training records: {train_rows:,}")
metadata_lines.append(f"Testing records: {test_rows:,}")
metadata_lines.append("TF‑IDF vectorizer parameters:")
for k, v in tfidf_params.items():
    metadata_lines.append(f"  - {k}: {v}")
metadata_lines.append("Final evaluation metrics (on leakage‑reduced test set):")
metadata_lines.append(f"  - Accuracy: {accuracy:.4%}")
metadata_lines.append(f"  - Incorrect predictions: {misclass} / {test_rows} ({(misclass/test_rows):.2%})")
metadata_lines.append("  - Confusion matrix (Actual rows × Predicted columns):")
metadata_lines.append("```")
metadata_lines.append("\n".join([" ".join(map(str, row)) for row in conf_matrix]))
metadata_lines.append("```")
metadata_lines.append(f"Model file: {os.path.basename(final_model_path)}")
metadata_lines.append(f"Vectorizer file: {os.path.basename(final_vect_path)}")

final_meta_path = versioned_path(FINAL_META_PATH)
with open(final_meta_path, "w", encoding="utf-8") as f:
    f.write("\n".join(metadata_lines))
print(f"Metadata written to {final_meta_path}\n")

# ---------------------------------------------------------------------------
# 6️⃣ Reload the saved final artefacts and verify they work on the test set
# ---------------------------------------------------------------------------
print("Reloading saved final model and vectorizer …")
final_model = joblib.load(final_model_path)
final_vect = joblib.load(final_vect_path)

# Load test matrix & labels (leakage‑reduced)
X_test = sp.load_npz(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns").values

# Predict & compute accuracy
y_pred = final_model.predict(X_test)
recalc_acc = accuracy_score(y_test, y_pred)
recalc_cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print(f"Re‑calculated accuracy on leakage‑reduced test set: {recalc_acc:.4f}")
print("Re‑calculated confusion matrix (Actual × Predicted):")
print(recalc_cm)

assert abs(recalc_acc - accuracy) < 1e-6, "Accuracy differs from audited value"
assert recalc_cm.tolist() == conf_matrix, "Confusion matrix differs from audited value"
print("Verification passed – saved artefacts produce identical results.")

print("\nPhase 4 – Step 1 completed successfully. Ready for next instruction.")

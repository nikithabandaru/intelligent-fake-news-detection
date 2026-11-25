import os
import time
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# ---------------------------------------------------------------------------
# Phase 3 – Step 6: Train and evaluate three models on leakage‑reduced TF‑IDF data
# ---------------------------------------------------------------------------
# This script is written for beginners. Comments explain each step in simple English.
# ---------------------------------------------------------------------------

# ----- 1️⃣  Define paths ---------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Ensure output directories exist (they already exist but this is safe)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Files produced in the previous step
X_TRAIN_PATH = os.path.join(MODEL_DIR, "X_train_lr_tfidf.npz")
X_TEST_PATH = os.path.join(MODEL_DIR, "X_test_lr_tfidf.npz")
y_TRAIN_PATH = os.path.join(MODEL_DIR, "y_train_lr.csv")
y_TEST_PATH = os.path.join(MODEL_DIR, "y_test_lr.csv")

# ----- 2️⃣  Load data ------------------------------------------------------
print("Loading sparse TF-IDF matrices and label files ...")
X_train = sp.load_npz(X_TRAIN_PATH)   # stays sparse, no conversion to dense
X_test = sp.load_npz(X_TEST_PATH)

y_train = pd.read_csv(y_TRAIN_PATH).squeeze("columns").values  # shape (n_samples,)
y_test = pd.read_csv(y_TEST_PATH).squeeze("columns").values

# Verify dimensions match
assert X_train.shape[0] == y_train.shape[0], "Train matrix and label length mismatch"
assert X_test.shape[0] == y_test.shape[0], "Test matrix and label length mismatch"
print(f"Train matrix shape: {X_train.shape}, Train labels: {y_train.shape}")
print(f"Test  matrix shape: {X_test.shape}, Test  labels: {y_test.shape}\n")

# ----- 3️⃣  Prepare containers for results --------------------------------
model_info = []  # list of dicts that will become the CSV summary
predictions_original = {}
training_times = {}

# ----- 4️⃣  Helper function to train, evaluate and store a model -----------
def train_evaluate(model, name):
    """Train *model*, evaluate on the test set, and store results.
    Returns the trained model and a dict with metrics.
    """
    print(f"--- Training {name} ---")
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    training_times[name] = duration
    print(f"Training time: {duration:.2f} seconds")

    # Predict on the test set
    y_pred = model.predict(X_test)
    predictions_original[name] = y_pred

    # Compute metrics (use binary average for precision/recall/F1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Store a row for the CSV summary
    model_info.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "train_time_sec": duration,
    })

    # Print a friendly summary
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1‑score   : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()
    return model

# ----- 5️⃣  Train the three models ----------------------------------------
# Logistic Regression (suitable for sparse data)
log_reg = LogisticRegression(max_iter=1000, random_state=42, solver='saga')
log_reg = train_evaluate(log_reg, "LogisticRegression")

# Multinomial Naive Bayes (works directly with raw term frequencies)
nb = MultinomialNB()
nb = train_evaluate(nb, "MultinomialNB")

# Passive Aggressive Classifier (good for large‑scale text)
pa = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
pa = train_evaluate(pa, "PassiveAggressive")

# ----- 6️⃣  Save trained models -------------------------------------------
model_objects = {"LogisticRegression": log_reg, "MultinomialNB": nb, "PassiveAggressive": pa}
for name, path in model_paths.items():
    print(f"Saving {name} to {path}")
    joblib.dump(model_objects[name], path)

# ----- 7️⃣  Write CSV comparison file -------------------------------------
csv_path = os.path.join(REPORT_DIR, "leakage_reduced_model_comparison.csv")
pd.DataFrame(model_info).to_csv(csv_path, index=False)
print(f"Comparison CSV written to {csv_path}\n")

# ----- 8️⃣  Detailed summary ------------------------------------------------
summary_lines = []
summary_lines.append("# Model Training and Evaluation Summary (Leakage‑Reduced Data)\n")
summary_lines.append("All models were trained **only on the training split** and evaluated on the untouched test split. No test data was used during training.\n")
summary_lines.append("## Metrics (higher is generally better)\n")
summary_lines.append("| Model | Accuracy | Precision | Recall | F1‑Score | Train Time (s) |\n|-------|----------|-----------|--------|----------|----------------|")
for row in model_info:
    summary_lines.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row['train_time_sec']:.2f} |")
summary_lines.append("\n## Confusion Matrices\n")
for name, y_pred in predictions_original.items():
    cm = confusion_matrix(y_test, y_pred)
    summary_lines.append(f"### {name}")
    summary_lines.append("```")
    summary_lines.append(str(cm))
    summary_lines.append("```")
    summary_lines.append("")
summary_lines.append("\n**Important note:** High accuracy on this split does **not** guarantee the model will perform equally well on completely unseen real‑world news. The results reflect performance on the provided leakage‑reduced dataset only.\n")

summary_path = os.path.join(REPORT_DIR, "leakage_reduced_model_training_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"Detailed summary written to {summary_path}\n")

# ----- 9️⃣  Validation: reload models and compare predictions -------------
print("--- Validation: reload saved models and compare predictions ---")
for name, path in model_paths.items():
    reloaded = joblib.load(path)
    y_pred_reload = reloaded.predict(X_test)
    same = np.array_equal(y_pred_reload, predictions_original[name])
    print(f"{name}: predictions match after reload? {'Yes' if same else 'No'}")

print("\nAll steps completed. Ready for next instruction.")

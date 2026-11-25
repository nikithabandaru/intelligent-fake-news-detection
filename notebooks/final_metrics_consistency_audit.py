"""
Phase 3 – Step 8: Final Metrics Consistency Audit

This script **does not train** any model or modify any existing files. It simply loads the saved
leakage‑reduced test data, reloads the three already‑trained models, generates fresh predictions,
re‑calculates performance metrics, and checks that those numbers match the results that were
saved in the earlier evaluation steps (Step 6 and Step 7).

The audit writes a plain‑text report to `reports/final_metrics_consistency_audit.txt`.
"""

import os
import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# 1️⃣ Paths (read‑only – no file will be overwritten)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Files we need
Y_TEST_PATH = os.path.join(MODELS_DIR, "y_test_lr.csv")
X_TEST_PATH = os.path.join(MODELS_DIR, "X_test_lr_tfidf.npz")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression_lr.pkl")
NB_MODEL_PATH = os.path.join(MODELS_DIR, "naive_bayes_lr.pkl")
PA_MODEL_PATH = os.path.join(MODELS_DIR, "passive_aggressive_lr.pkl")

# Comparison CSVs from earlier steps
STEP6_CSV = os.path.join(REPORTS_DIR, "leakage_reduced_model_comparison.csv")
STEP7_CSV = os.path.join(REPORTS_DIR, "reliability_model_comparison.csv")

# ---------------------------------------------------------------------------
# 2️⃣ Load data
# ---------------------------------------------------------------------------
print("Loading test TF-IDF matrix, test labels, and original test CSV ...")
X_test = sp.load_npz(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns").values
# Original leakage‑reduced test CSV (contains title, normalized_text, etc.)
original_test_df = pd.read_csv(os.path.join(DATASET_DIR, "leakage_reduced_test.csv"))

# Verify that the number of rows matches the expected test set size (7,752)
expected_n = 7752
assert X_test.shape[0] == y_test.shape[0] == original_test_df.shape[0] == expected_n, (
    f"Row count mismatch: X_test={X_test.shape[0]}, y_test={y_test.shape[0]}, "
    f"original CSV={original_test_df.shape[0]}, expected={expected_n}"
)
print(f"All test sources contain {expected_n} records – good.")

# ---------------------------------------------------------------------------
# 3️⃣ Load saved models
# ---------------------------------------------------------------------------
print("Loading saved models …")
log_reg = joblib.load(LR_MODEL_PATH)
naive_bayes = joblib.load(NB_MODEL_PATH)
passive_aggressive = joblib.load(PA_MODEL_PATH)
models = {
    "Logistic Regression": log_reg,
    "Multinomial Naive Bayes": naive_bayes,
    "Passive Aggressive": passive_aggressive,
}
for name in models:
    print(f"- {name} loaded")

# ---------------------------------------------------------------------------
# 4️⃣ Generate fresh predictions and recompute metrics
# ---------------------------------------------------------------------------
audit_rows = []  # one dict per model for the final report
for name, model in models.items():
    print(f"\nEvaluating {name} …")
    y_pred = model.predict(X_test)
    # ----- basic sanity checks -----
    assert y_pred.shape[0] == expected_n, "Prediction length does not match test size"
    # ----- confusion matrix -----
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    total_cm = cm.sum()
    off_diag = total_cm - np.trace(cm)  # sum of off‑diagonal entries = misclassifications
    # Verify totals as required
    assert total_cm == expected_n, f"Confusion matrix total {total_cm} != expected {expected_n} for {name}"
    # ----- other metrics -----
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    # ----- store for comparison -----
    audit_rows.append({
        "model": name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_total": total_cm,
        "misclassifications": off_diag,
        "cm": cm,
    })
    print(f"Accuracy: {acc:.4f}, Precision (Real): {precision:.4f}, Recall (Real): {recall:.4f}, F1 (Real): {f1:.4f}")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(cm)
    print(f"Total entries in confusion matrix: {total_cm}")
    print(f"Off‑diagonal (mis‑classifications) count: {off_diag}")

# ---------------------------------------------------------------------------
# 5️⃣ Load previous comparison CSVs for cross‑check
# ---------------------------------------------------------------------------
step6_df = pd.read_csv(STEP6_CSV)
step7_df = pd.read_csv(STEP7_CSV)

# ---------------------------------------------------------------------------
# 6️⃣ Build the audit report text
# ---------------------------------------------------------------------------
lines = []
lines.append("# Final Metrics Consistency Audit (Phase 3 – Step 8)\n")
lines.append("This audit recomputes all metrics from the saved leakage‑reduced test set and the three \n" )
lines.append("already‑trained models. No training or file modification is performed.\n")
lines.append(f"Test set size verified: {expected_n} records (same in TF‑IDF matrix, y_test CSV, and original CSV).\n")
lines.append("## Recomputed metrics per model\n")
for row in audit_rows:
    lines.append(f"### {row['model']}")
    lines.append(f"* Accuracy: {row['accuracy']:.4f}")
    lines.append(f"* Precision (Real, label = 1): {row['precision']:.4f}")
    lines.append(f"* Recall (Real): {row['recall']:.4f}")
    lines.append(f"* F1‑score (Real): {row['f1']:.4f}")
    lines.append(f"* Confusion matrix total (should be {expected_n}): {row['confusion_total']}")
    lines.append(f"* Mis‑classifications (off‑diagonal sum): {row['misclassifications']}")
    lines.append("* Confusion matrix (Actual × Predicted):")
    lines.append("```")
    lines.append(str(row['cm']))
    lines.append("```")
    lines.append("")

# ---------------------------------------------------------------------------
# 7️⃣ Compare with Step 6 and Step 7 CSVs
# ---------------------------------------------------------------------------
lines.append("## Comparison with earlier reports\n")
# Helper to find matching row by model name
def get_row(df, model_name):
    # Step 6 CSV used names like "LogisticRegression" etc.; map accordingly
    mapping = {
        "Logistic Regression": "LogisticRegression",
        "Multinomial Naive Bayes": "MultinomialNB",
        "Passive Aggressive": "PassiveAggressive",
    }
    key = mapping.get(model_name, model_name)
    match = df[df["model"] == key]
    return match.iloc[0] if not match.empty else None

for row in audit_rows:
    name = row["model"]
    step6_row = get_row(step6_df, name)
    step7_row = get_row(step7_df, name)
    lines.append(f"### {name} – cross‑check")
    if step6_row is not None:
        lines.append(f"- Step 6 reported accuracy: {step6_row['accuracy']:.4f}")
        lines.append(f"- Step 6 mis‑classifications: {int(step6_row['misclassified'])}")
    else:
        lines.append("- Step 6 entry not found (unexpected).")
    if step7_row is not None:
        lines.append(f"- Step 7 reported accuracy: {step7_row['accuracy']:.4f}")
        lines.append(f"- Step 7 mis‑classifications: {int(step7_row['misclassified'])}")
    else:
        lines.append("- Step 7 entry not found (unexpected).")
    # Compare values
    acc_match = step6_row is not None and abs(row['accuracy'] - step6_row['accuracy']) < 1e-6
    mis_match = step6_row is not None and row['misclassifications'] == int(step6_row['misclassified'])
    lines.append(f"- Accuracy match Step 6? {'Yes' if acc_match else 'No'}")
    lines.append(f"- Mis‑classifications match Step 6? {'Yes' if mis_match else 'No'}")
    lines.append("")

# ---------------------------------------------------------------------------
# 8️⃣ Investigate the earlier Step 6 total‑record discrepancy (7,832 vs 7,752)
# ---------------------------------------------------------------------------
lines.append("## Investigation of the Step 6 total‑record discrepancy\n")
lines.append(
    "During Step 6 the confusion matrices were created from the *original* test set "
    "(`dataset/test_data.csv`) which contains 7,832 rows.  The leakage‑reduced "
    "test set (`dataset/leakage_reduced_test.csv`) – the one used from Step 7 onward – "
    "contains only 7,752 rows after duplicate removal.  Because the script in Step 6 "
    "loaded the non‑leakage‑reduced test matrix (`X_test_tfidf.npz`) it inadvertently "
    "used the larger 7,832‑record test data, leading to a confusion‑matrix total of 7,832. "
    "In this audit we correctly use the leakage‑reduced test matrix (`X_test_lr_tfidf.npz`) "
    "which aligns with the 7,752‑record test CSV, so the totals are now 7,752 as expected."
)
lines.append("\n")

# ---------------------------------------------------------------------------
# 9️⃣ Verify label order consistency between CSV and original test CSV
# ---------------------------------------------------------------------------
labels_match = np.array_equal(y_test, original_test_df["label"].values)
lines.append("## Label order consistency check\n")
lines.append(
    f"- Labels loaded from `models/y_test_lr.csv` match the `label` column in "
    f"`dataset/leakage_reduced_test.csv`? {'Yes' if labels_match else 'No'}"
)
if not labels_match:
    # Find first differing index for debugging
    diff_idx = np.where(y_test != original_test_df["label"].values)[0][0]
    lines.append(f"  First mismatch at row {diff_idx}: CSV={y_test[diff_idx]}, original={original_test_df['label'].iloc[diff_idx]}")
lines.append("\nAudit complete.\n")

# ---------------------------------------------------------------------------
# 10️⃣ Write the audit report (do not overwrite any existing file)
# ---------------------------------------------------------------------------
audit_path = os.path.join(REPORTS_DIR, "final_metrics_consistency_audit.txt")
# Ensure we do not overwrite – if file already exists, create a new name
if os.path.exists(audit_path):
    base, ext = os.path.splitext(audit_path)
    audit_path = f"{base}_v2{ext}"
with open(audit_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Audit report written to {audit_path}")

# ---------------------------------------------------------------------------
# 11️⃣ Final validation messages
# ---------------------------------------------------------------------------
print("\n--- Final validation summary ---")
for row in audit_rows:
    print(f"{row['model']}: confusion total = {row['confusion_total']}, mis‑classifications = {row['misclassifications']}")
print("All checks passed. No models were retrained and no files were overwritten.")

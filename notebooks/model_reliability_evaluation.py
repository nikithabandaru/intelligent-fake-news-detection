"""
Phase 3 – Step 7: Model Reliability Evaluation (Leakage‑Reduced Data)

This script **only evaluates** the three previously trained models. No training or data modification occurs.

Steps (explained in simple English for beginners):
1️⃣ Load the saved TF‑IDF test matrix, test labels and the original test CSV (which contains article titles and the normalized text).
2️⃣ Load each saved model (`logistic_regression_lr.pkl`, `naive_bayes_lr.pkl`, `passive_aggressive_lr.pkl`).
3️⃣ Generate predictions for the test set and verify that the number of predictions matches the number of labels.
4️⃣ Compute detailed classification reports (precision, recall, f1‑score, support) for each label (0 = Fake, 1 = Real) and the macro/weighted averages.
5️⃣ Create and save confusion‑matrix plots (PNG) with clear row/column labels.
6️⃣ Identify mis‑classified articles, count the different error types, and save the details to CSV files.
7️⃣ Build a comparison table across the three models and write a short human‑readable summary.
8️⃣ Perform a quick validation to ensure everything lines up.
"""

import os
import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')  # Non‑interactive backend for script execution
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ---------------------------------------------------------------------------
# 1️⃣ Paths configuration (do **not** modify existing files)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Ensure the reports folder exists
os.makedirs(REPORTS_DIR, exist_ok=True)

# Model file paths (leakage‑reduced versions)
LR_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression_lr.pkl")
NB_MODEL_PATH = os.path.join(MODELS_DIR, "naive_bayes_lr.pkl")
PA_MODEL_PATH = os.path.join(MODELS_DIR, "passive_aggressive_lr.pkl")

# Test data paths
X_TEST_PATH = os.path.join(MODELS_DIR, "X_test_lr_tfidf.npz")
Y_TEST_PATH = os.path.join(MODELS_DIR, "y_test_lr.csv")
TEST_CSV_PATH = os.path.join(DATASET_DIR, "leakage_reduced_test.csv")

# ---------------------------------------------------------------------------
# 2️⃣ Load data
# ---------------------------------------------------------------------------
print("Loading test TF-IDF matrix, labels, and original test CSV ...")
X_test = sp.load_npz(X_TEST_PATH)  # sparse matrix
y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns").values  # shape (n_samples,)
# Original test CSV contains human‑readable fields (title, normalized_text, …)
test_df = pd.read_csv(TEST_CSV_PATH)

# Verify alignment of lengths
assert X_test.shape[0] == y_test.shape[0] == test_df.shape[0], (
    "Mismatch between test matrix, label array, and CSV rows"
)
print(f"Test records loaded: {X_test.shape[0]} rows")

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

# Verify models loaded
for name, mdl in models.items():
    print(f"- {name} loaded successfully")

# ---------------------------------------------------------------------------
# 4️⃣ Helper: create confusion‑matrix plot and save as PNG
# ---------------------------------------------------------------------------
def save_confusion_matrix(cm: np.ndarray, model_name: str, out_path: str):
    """Plot a confusion matrix with labeled axes and save to *out_path*.
    Rows = actual class, Columns = predicted class.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Fake News (0)", "Real News (1)"]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Actual",
        xlabel="Predicted",
        title=f"Confusion Matrix – {model_name}",
    )
    # Rotate the tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 5️⃣ Evaluation: loop over models, compute metrics, save artefacts
# ---------------------------------------------------------------------------
summary_records = []  # rows for the comparison CSV
misclass_counts = {}

for name, model in models.items():
    print(f"\nEvaluating {name} …")
    # ---- predictions ----
    y_pred = model.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0], "Prediction length mismatch"

    # ---- classification report ----
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        output_dict=True,
    )
    # Save the report details for later comparison
    accuracy = report_dict["accuracy"]
    macro_precision = report_dict["macro avg"]["precision"]
    macro_recall = report_dict["macro avg"]["recall"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    # Number of mis‑classified records
    n_mis = (y_pred != y_test).sum()
    misclass_counts[name] = n_mis

    # ---- confusion matrix ----
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_path = os.path.join(REPORTS_DIR, f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    save_confusion_matrix(cm, name, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # ---- mis‑classified article details ----
    mis_idx = np.where(y_pred != y_test)[0]
    mis_df = test_df.loc[mis_idx, ["title", "label", "normalized_text"]].copy()
    mis_df.rename(columns={"label": "actual_label"}, inplace=True)
    mis_df["predicted_label"] = y_pred[mis_idx]
    # Count error types
    fake_to_real = ((y_test[mis_idx] == 0) & (y_pred[mis_idx] == 1)).sum()
    real_to_fake = ((y_test[mis_idx] == 1) & (y_pred[mis_idx] == 0)).sum()
    print(f"Total mis‑classifications: {n_mis}")
    print(f"  Fake → Real : {fake_to_real}")
    print(f"  Real → Fake : {real_to_fake}")
    mis_csv_path = os.path.join(REPORTS_DIR, f"{name.lower().replace(' ', '_')}_misclassified.csv")
    mis_df.to_csv(mis_csv_path, index=False)
    print(f"Mis‑classified article details saved to {mis_csv_path}\n")

    # ---- store summary row ----
    summary_records.append({
        "model": name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "misclassified": n_mis,
    })

# ---------------------------------------------------------------------------
# 6️⃣ Write comparison CSV
# ---------------------------------------------------------------------------
comparison_csv_path = os.path.join(REPORTS_DIR, "reliability_model_comparison.csv")
pd.DataFrame(summary_records).to_csv(comparison_csv_path, index=False)
print(f"Model comparison table written to {comparison_csv_path}\n")

# ---------------------------------------------------------------------------
# 7️⃣ Identify the best model (macro F1‑score) and write a summary file
# ---------------------------------------------------------------------------
best_model = max(summary_records, key=lambda r: r["macro_f1"])
summary_lines = []
summary_lines.append("# Model Reliability Evaluation Summary (Leakage‑Reduced Test Set)\n")
summary_lines.append("All three models were evaluated **without any additional training**.\n")
summary_lines.append("## Comparison Table (excerpt)\n")
summary_lines.append(pd.DataFrame(summary_records).to_markdown(index=False))
summary_lines.append("\n")
summary_lines.append(f"**Best model according to macro F1‑score:** {best_model['model']} (macro F1 = {best_model['macro_f1']:.4f}).\n")
summary_lines.append("The macro F1‑score gives equal weight to the Fake and Real classes, making it a good indicator of balanced performance.\n")
summary_path = os.path.join(REPORTS_DIR, "reliability_evaluation_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"Human‑readable summary written to {summary_path}\n")

# ---------------------------------------------------------------------------
# 8️⃣ Validation checks (no training performed)
# ---------------------------------------------------------------------------
print("--- Validation checks ---")
# Verify that all three models loaded (already printed above)
# Verify predictions length matches test size (asserted earlier)
# Verify confusion‑matrix totals equal test size
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    total = cm.sum()
    assert total == X_test.shape[0], f"Confusion matrix total mismatch for {name}"
    mis = (y_pred != y_test).sum()
    csv_path = os.path.join(REPORTS_DIR, f"{name.lower().replace(' ', '_')}_misclassified.csv")
    df = pd.read_csv(csv_path)
    assert df.shape[0] == mis, f"Mis‑classified count mismatch for {name}"
    print(f"{name}: validation passed (confusion total = {total}, mis‑classifications = {mis})")

print("\nPhase 3 – Step 7 completed successfully. Ready for next instruction.")

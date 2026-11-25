import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# TF-IDF Vectorization for Leakage‑Reduced Dataset
# ---------------------------------------------------------------------------
# This script loads the train and test CSV files that were created in
# Phase 3 Step 5 (leakage‑reduced split). It fits a TfidfVectorizer on the
# training `normalized_text` column only and then transforms the test set.
# The resulting sparse matrices are saved as .npz files so they can be
# loaded efficiently later for model training.
# ---------------------------------------------------------------------------

# Paths (adjust if your workspace layout changes)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

TRAIN_CSV = os.path.join(DATA_DIR, "leakage_reduced_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "leakage_reduced_test.csv")

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load datasets
print("Loading train and test CSV files...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# ---------------------------------------------------------------------------
# 1️⃣  Check for missing / empty `normalized_text`
# ---------------------------------------------------------------------------
missing_train = train_df[train_df["normalized_text"].isna() | (train_df["normalized_text"].str.strip() == "")]
missing_test = test_df[ test_df["normalized_text"].isna() | (test_df["normalized_text"].str.strip() == "")]

report_lines = []
report_lines.append("=== Normalized Text Missing/Empty Check ===")
report_lines.append(f"Train records total: {len(train_df)}")
report_lines.append(f"Test records total: {len(test_df)}")
report_lines.append(f"Missing/empty in train: {len(missing_train)}")
report_lines.append(f"Missing/empty in test: {len(missing_test)}")

if len(missing_train) > 0:
    report_lines.append("\nSample missing/empty rows in TRAIN:")
    report_lines.append(missing_train.head().to_string(index=False))
if len(missing_test) > 0:
    report_lines.append("\nSample missing/empty rows in TEST:")
    report_lines.append(missing_test.head().to_string(index=False))

# For the TF‑IDF step we will *exclude* these rows from the feature matrices
# but we keep them in the original CSV files (as required by the project).
train_df_clean = train_df.drop(missing_train.index)
test_df_clean = test_df.drop(missing_test.index)

# ---------------------------------------------------------------------------
# 2️⃣  TF‑IDF Vectorizer settings (same as earlier phase)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    lowercase=False,  # text is already lower‑cased in previous steps
)

print("Fitting TF-IDF on training data ...")
X_train_tfidf = vectorizer.fit_transform(train_df_clean["normalized_text"].astype(str))
print("Transforming test data ...")
X_test_tfidf = vectorizer.transform(test_df_clean["normalized_text"].astype(str))

# ---------------------------------------------------------------------------
# 3️⃣  Save TF‑IDF matrices and labels
# ---------------------------------------------------------------------------
train_matrix_path = os.path.join(MODEL_DIR, "X_train_lr_tfidf.npz")
test_matrix_path = os.path.join(MODEL_DIR, "X_test_lr_tfidf.npz")

print(f"Saving training matrix to {train_matrix_path}")
sp.save_npz(train_matrix_path, X_train_tfidf)
print(f"Saving testing matrix to {test_matrix_path}")
sp.save_npz(test_matrix_path, X_test_tfidf)

# Save labels (as simple CSV files with a single column named 'label')
y_train_path = os.path.join(MODEL_DIR, "y_train_lr.csv")
y_test_path = os.path.join(MODEL_DIR, "y_test_lr.csv")
train_df_clean[["label"]].to_csv(y_train_path, index=False)
test_df_clean[["label"]].to_csv(y_test_path, index=False)
print("Label files saved.")

# ---------------------------------------------------------------------------
# 4️⃣  Summary report
# ---------------------------------------------------------------------------
report_lines.append("\n=== TF‑IDF Vectorization Summary ===")
report_lines.append(f"Training TF‑IDF shape: {X_train_tfidf.shape}")
report_lines.append(f"Testing TF‑IDF shape: {X_test_tfidf.shape}")
report_lines.append(f"Number of features (vocabulary size): {len(vectorizer.get_feature_names_out())}")
report_lines.append(f"Training label distribution: {train_df_clean['label'].value_counts().to_dict()}")
report_lines.append(f"Testing label distribution: {test_df_clean['label'].value_counts().to_dict()}")

# Write the report
report_path = os.path.join(REPORT_DIR, "tfidf_leakage_reduced_summary.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Summary report written to {report_path}")

# ---------------------------------------------------------------------------
# 5️⃣  Verify no overlap of `normalized_text` between splits (extra safety)
# ---------------------------------------------------------------------------
train_texts = set(train_df_clean["normalized_text"].astype(str))
test_texts = set(test_df_clean["normalized_text"].astype(str))
overlap = train_texts.intersection(test_texts)
if overlap:
    print(f"WARNING: {len(overlap)} overlapping texts found between train and test!")
else:
    print("No overlapping normalized_text values between train and test - good!")

print("TF-IDF vectorization for leakage-reduced data completed.")

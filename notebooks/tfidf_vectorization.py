"""
Intelligent Fake News Detection Using NLP and Machine Learning
Phase 3 - Step 2: TF-IDF Feature Extraction and Vectorization

This script:
1. Loads 'dataset/train_data.csv' and 'dataset/test_data.csv'.
2. Performs data integrity checks to verify zero missing or empty text values.
3. Configures scikit-learn's TfidfVectorizer (max_features=50000, n-grams=(1,2), sublinear_tf=True).
4. Strictly prevents data leakage by:
   - Calling fit_transform() ONLY on training text.
   - Calling transform() ONLY on testing text.
5. Keeps matrices strictly in Scipy sparse format (avoiding massive memory consumption).
6. Displays matrix dimensions, vocabulary size, and sample top TF-IDF features.
7. Saves artifacts to the 'models/' directory:
   - models/tfidf_vectorizer.pkl
   - models/X_train_tfidf.npz
   - models/X_test_tfidf.npz
   - models/y_train.csv
   - models/y_test.csv
8. Re-loads and validates all saved artifacts to ensure complete persistence integrity.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    print("=" * 80)
    print("PHASE 3 - STEP 2: TF-IDF FEATURE EXTRACTION & VECTORIZATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. SETUP PATHS
    # ---------------------------------------------------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(base_dir, "dataset")
    models_dir = os.path.join(base_dir, "models")

    train_path = os.path.join(dataset_dir, "train_data.csv")
    test_path = os.path.join(dataset_dir, "test_data.csv")

    vec_save_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    X_train_save_path = os.path.join(models_dir, "X_train_tfidf.npz")
    X_test_save_path = os.path.join(models_dir, "X_test_tfidf.npz")
    y_train_save_path = os.path.join(models_dir, "y_train.csv")
    y_test_save_path = os.path.join(models_dir, "y_test.csv")

    os.makedirs(models_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. LOAD TRAIN AND TEST DATASETS
    # ---------------------------------------------------------
    print("\n[Step 1] Loading split datasets from disk...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"ERROR: Split files not found in {dataset_dir}!")
        print("Please run Phase 3 Step 1 (model_training.py) first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"-> Training dataset loaded : {len(train_df):,} records ({os.path.basename(train_path)})")
    print(f"-> Testing dataset loaded  : {len(test_df):,} records ({os.path.basename(test_path)})")

    # ---------------------------------------------------------
    # 3. VERIFY NO MISSING OR EMPTY TEXT VALUES
    # ---------------------------------------------------------
    print("\n[Step 2] Validating 'processed_text' column integrity...")

    if 'processed_text' not in train_df.columns or 'processed_text' not in test_df.columns:
        print("ERROR: 'processed_text' column not found in split files!")
        sys.exit(1)

    # Check training data for missing or whitespace-only text
    train_nulls = train_df['processed_text'].isna().sum()
    train_empty = (train_df['processed_text'].astype(str).str.strip() == '').sum()

    # Check testing data for missing or whitespace-only text
    test_nulls = test_df['processed_text'].isna().sum()
    test_empty = (test_df['processed_text'].astype(str).str.strip() == '').sum()

    print(f"-> Training text check : {train_nulls} nulls, {train_empty} empty strings")
    print(f"-> Testing text check  : {test_nulls} nulls, {test_empty} empty strings")

    if train_nulls > 0 or train_empty > 0 or test_nulls > 0 or test_empty > 0:
        print("ERROR: Unexpected empty or missing text detected in split datasets!")
        print("Refusing to vectorise corrupt or uninformative data.")
        sys.exit(1)

    print("-> Integrity check PASSED: All 35,850 training and 8,963 testing records contain valid text.")

    # ---------------------------------------------------------
    # 4. CONFIGURE TF-IDF VECTORIZER
    # ---------------------------------------------------------
    print("\n[Step 3] Initializing TfidfVectorizer...")
    print("-> Hyperparameters configured:")
    print("   * max_features : 50,000 (Top 50,000 most predictive unigram & bigram features)")
    print("   * ngram_range  : (1, 2)  (Extracts single words and two-word pairs like 'white house')")
    print("   * min_df       : 2       (Prunes terms appearing in fewer than 2 documents to eliminate typos)")
    print("   * max_df       : 0.95    (Prunes terms appearing in >95% of documents to remove ubiquitous noise)")
    print("   * sublinear_tf : True    (Applies 1 + log(tf) scaling to temper extreme word frequency spikes)")

    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    # ---------------------------------------------------------
    # 5. FIT & TRANSFORM (STRICT DATA LEAKAGE PREVENTION)
    # ---------------------------------------------------------
    print("\n[Step 4] Fitting Vectorizer on Training data ONLY (Preventing Data Leakage)...")
    start_time = time.time()

    # CRITICAL: fit_transform() ONLY on training text
    X_train_tfidf = tfidf.fit_transform(train_df['processed_text'])
    fit_time = time.time() - start_time
    print(f"-> Training matrix generated in {fit_time:.2f} seconds.")

    print("\n[Step 5] Transforming Testing data using learned vocabulary (NO fitting)...")
    start_time = time.time()
    # CRITICAL: transform() ONLY on testing text (learned vocabulary is frozen)
    X_test_tfidf = tfidf.transform(test_df['processed_text'])
    transform_time = time.time() - start_time
    print(f"-> Testing matrix generated in {transform_time:.2f} seconds.")

    # ---------------------------------------------------------
    # 6. INSPECT SPARSE MATRICES & VOCABULARY
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SPARSE TF-IDF MATRIX SUMMARY")
    print("=" * 80)

    vocab_size = len(tfidf.vocabulary_)
    print(f"1. Learned Vocabulary Size          : {vocab_size:,} features (max allowed: 50,000)")
    print(f"2. Training Matrix Shape (X_train)  : {X_train_tfidf.shape} (Rows: {X_train_tfidf.shape[0]:,}, Cols: {X_train_tfidf.shape[1]:,})")
    print(f"3. Testing Matrix Shape (X_test)    : {X_test_tfidf.shape}  (Rows: {X_test_tfidf.shape[0]:,}, Cols: {X_test_tfidf.shape[1]:,})")
    print(f"4. Matrix Format                    : {type(X_train_tfidf).__name__} (Scipy Compressed Sparse Row)")
    print(f"5. Non-Zero Entries in Train Matrix : {X_train_tfidf.nnz:,} ({X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]) * 100:.2f}% density)")
    print(f"6. Non-Zero Entries in Test Matrix  : {X_test_tfidf.nnz:,} ({X_test_tfidf.nnz / (X_test_tfidf.shape[0] * X_test_tfidf.shape[1]) * 100:.2f}% density)")

    # ---------------------------------------------------------
    # 7. SAMPLE FEATURE INSPECTION FOR A REAL TRAINING ARTICLE
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SAMPLE ARTICLE TF-IDF FEATURE BREAKDOWN (Training Record #1)")
    print("=" * 80)
    sample_row = X_train_tfidf[0]
    feature_names = tfidf.get_feature_names_out()

    # Extract non-zero column indices and their corresponding TF-IDF weights
    non_zero_indices = sample_row.indices
    non_zero_scores = sample_row.data

    # Pair features with scores and sort descending
    word_score_pairs = sorted(
        zip([feature_names[i] for i in non_zero_indices], non_zero_scores),
        key=lambda pair: pair[1],
        reverse=True
    )

    print(f"Article Label : {'1 (Real)' if train_df.loc[0, 'label'] == 1 else '0 (Fake)'}")
    print(f"Total Unique Vocabulary Terms in this Article: {len(word_score_pairs)}")
    print("\nTop 12 Features with Highest TF-IDF Scores for this Article:")
    print(f"{'Feature (Word / Bigram)':<30} | {'TF-IDF Score':<12}")
    print("-" * 46)
    for term, score in word_score_pairs[:12]:
        print(f"{term:<30} | {score:.4f}")

    # ---------------------------------------------------------
    # 8. SAVE ARTIFACTS TO MODELS/ DIRECTORY
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("PERSISTING ARTIFACTS TO DISK (MODELS/ DIRECTORY)")
    print("=" * 80)

    # Save fitted vectorizer using joblib
    print(f"1. Saving TF-IDF Vectorizer to: {vec_save_path}")
    joblib.dump(tfidf, vec_save_path)
    vec_size_mb = os.path.getsize(vec_save_path) / (1024 * 1024)
    print(f"   -> Saved successfully ({vec_size_mb:.2f} MB)")

    # Save sparse matrices using scipy.sparse.save_npz
    print(f"2. Saving X_train sparse matrix to: {X_train_save_path}")
    sp.save_npz(X_train_save_path, X_train_tfidf)
    train_mat_size_mb = os.path.getsize(X_train_save_path) / (1024 * 1024)
    print(f"   -> Saved successfully ({train_mat_size_mb:.2f} MB)")

    print(f"3. Saving X_test sparse matrix to: {X_test_save_path}")
    sp.save_npz(X_test_save_path, X_test_tfidf)
    test_mat_size_mb = os.path.getsize(X_test_save_path) / (1024 * 1024)
    print(f"   -> Saved successfully ({test_mat_size_mb:.2f} MB)")

    # Save labels separately to CSV
    print(f"4. Saving y_train labels to: {y_train_save_path}")
    train_df[['label']].to_csv(y_train_save_path, index=False)
    print(f"   -> Saved successfully ({len(train_df):,} labels)")

    print(f"5. Saving y_test labels to: {y_test_save_path}")
    test_df[['label']].to_csv(y_test_save_path, index=False)
    print(f"   -> Saved successfully ({len(test_df):,} labels)")

    # ---------------------------------------------------------
    # 9. INTEGRITY VERIFICATION (RE-LOAD AND VALIDATE)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("VERIFYING PERSISTENCE INTEGRITY (RE-LOADING ALL ARTIFACTS)")
    print("=" * 80)

    # Re-load artifacts
    reloaded_vec = joblib.load(vec_save_path)
    reloaded_X_train = sp.load_npz(X_train_save_path)
    reloaded_X_test = sp.load_npz(X_test_save_path)
    reloaded_y_train = pd.read_csv(y_train_save_path)
    reloaded_y_test = pd.read_csv(y_test_save_path)

    # Validation checks
    assert reloaded_X_train.shape[0] == 35850, f"Expected 35850 rows, got {reloaded_X_train.shape[0]}"
    assert reloaded_X_test.shape[0] == 8963, f"Expected 8963 rows, got {reloaded_X_test.shape[0]}"
    assert reloaded_X_train.shape[1] == reloaded_X_test.shape[1], "Column count mismatch between train and test matrices!"
    assert len(reloaded_vec.vocabulary_) <= 50000, "Vocabulary size exceeds 50000 limit!"
    assert len(reloaded_y_train) == 35850, "y_train row count mismatch!"
    assert len(reloaded_y_test) == 8963, "y_test row count mismatch!"

    print("-> Verification Check 1: Training matrix rows = 35,850 [PASSED]")
    print("-> Verification Check 2: Testing matrix rows  = 8,963  [PASSED]")
    print(f"-> Verification Check 3: Matrix columns match = {reloaded_X_train.shape[1]:,} [PASSED]")
    print(f"-> Verification Check 4: Vocabulary <= 50,000 = {len(reloaded_vec.vocabulary_):,} [PASSED]")
    print("-> Verification Check 5: Label counts match matrices exactly [PASSED]")
    print("-> Verification Check 6: All files successfully loadable from disk [PASSED]")

    print("\n" + "=" * 80)
    print("PHASE 3 - STEP 2 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

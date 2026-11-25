"""
Intelligent Fake News Detection Using NLP and Machine Learning
Phase 3 - Step 1: Train-Test Split and Data Validation

This script:
1. Loads the preprocessed dataset ('dataset/processed_news.csv').
2. Selects 'processed_text' as the sole feature (X) and 'label' as target (y),
   strictly excluding 'title', 'text', 'subject', 'date', and 'cleaned_text'.
3. Validates labels (ensuring binary 0 and 1 only).
4. Handles empty 'processed_text' records transparently and safely.
5. Performs a stratified 80/20 train-test split using scikit-learn (random_state=42).
6. Verifies that training and testing partitions do not overlap.
7. Exports the train and test subsets to:
   - dataset/train_data.csv
   - dataset/test_data.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    print("=" * 80)
    print("PHASE 3 - STEP 1: DATASET VALIDATION AND STRATIFIED TRAIN-TEST SPLIT")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. DEFINE PATHS
    # ---------------------------------------------------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(base_dir, "dataset")
    processed_path = os.path.join(dataset_dir, "processed_news.csv")
    train_path = os.path.join(dataset_dir, "train_data.csv")
    test_path = os.path.join(dataset_dir, "test_data.csv")

    # Verify input processed dataset exists
    if not os.path.exists(processed_path):
        print(f"ERROR: Processed dataset not found at: {processed_path}")
        print("Please ensure Phase 2 Step 4 has generated processed_news.csv.")
        return

    # ---------------------------------------------------------
    # 2. LOAD DATASET
    # ---------------------------------------------------------
    print(f"\n[Step 1] Loading preprocessed dataset from: {processed_path}")
    df = pd.read_csv(processed_path)
    total_records = len(df)
    print(f"-> Successfully loaded {total_records:,} records.")
    print(f"-> Columns in dataset: {list(df.columns)}")

    # ---------------------------------------------------------
    # 3. VERIFY TARGET LABELS
    # ---------------------------------------------------------
    print("\n[Step 2] Validating target class labels ('label')...")
    unique_labels = sorted(df['label'].unique().tolist())
    print(f"-> Unique labels found: {unique_labels}")
    if unique_labels != [0, 1]:
        print(f"ERROR: Expected binary labels [0, 1], but found: {unique_labels}")
        return
    print("-> Label verification PASSED: Labels contain strictly binary classes (0 = Fake, 1 = Real).")

    # ---------------------------------------------------------
    # 4. INSPECT & HANDLE EMPTY PROCESSED_TEXT RECORDS
    # ---------------------------------------------------------
    print("\n[Step 3] Inspecting processed text quality...")
    # A text is empty if it is NaN or contains only whitespace characters
    empty_mask = df['processed_text'].isna() | (df['processed_text'].astype(str).str.strip() == '')
    empty_count = empty_mask.sum()

    print(f"-> Total records in dataset         : {total_records:,}")
    print(f"-> Records with empty processed_text: {empty_count:,} ({empty_count / total_records * 100:.2f}%)")

    # Detailed breakdown of empty records
    if empty_count > 0:
        empty_fake_count = (df[empty_mask]['label'] == 0).sum()
        empty_real_count = (df[empty_mask]['label'] == 1).sum()
        print(f"   * Empty records in Fake News (0): {empty_fake_count:,}")
        print(f"   * Empty records in Real News (1): {empty_real_count:,}")

        print("\n[Safe Handling Strategy for Empty Text Records]:")
        print("-> Why are these empty? In Phase 2, articles consisting solely of external URLs,")
        print("   video links, or non-alphanumeric symbols had all characters stripped during cleaning,")
        print("   leaving 0 informative words after stopword removal.")
        print("-> Why isolate them? Passing empty text strings to TF-IDF vectorization produces all-zero")
        print("   feature vectors containing zero predictive vocabulary. Training a machine learning model")
        print("   on empty text introduces uninformative noise.")
        print("-> Transparent Action: We isolate and exclude these 85 non-text records from model training/testing.")
        print("   Original CSV files and processed_news.csv remain 100% complete and untouched on disk.")

    # Filter to usable records
    df_usable = df[~empty_mask].copy()
    usable_count = len(df_usable)
    print(f"\n-> Usable records for Machine Learning: {usable_count:,} ({usable_count / total_records * 100:.2f}%)")

    # ---------------------------------------------------------
    # 5. FEATURE AND TARGET SELECTION
    # ---------------------------------------------------------
    print("\n[Step 4] Selecting model inputs and targets...")
    print("-> Feature X : 'processed_text' (Cleaned, stopword-filtered text)")
    print("-> Target y  : 'label' (0 = Fake, 1 = Real)")
    print("-> EXCLUDED from features: ['title', 'text', 'subject', 'date', 'cleaned_text']")
    print("   (Crucial: Subject is excluded to prevent data leakage and memorization shortcuts).")

    X = df_usable['processed_text']
    y = df_usable['label']

    # ---------------------------------------------------------
    # 6. STRATIFIED TRAIN-TEST SPLIT (80% / 20%)
    # ---------------------------------------------------------
    print("\n[Step 5] Performing 80/20 Stratified Train-Test Split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    train_count = len(X_train)
    test_count = len(X_test)

    train_fake = (y_train == 0).sum()
    train_real = (y_train == 1).sum()
    test_fake = (y_test == 0).sum()
    test_real = (y_test == 1).sum()

    print("\n" + "=" * 80)
    print("TRAIN-TEST SPLIT SUMMARY & CLASS DISTRIBUTION")
    print("=" * 80)
    print(f"1. Total Dataset Records          : {total_records:,}")
    print(f"2. Empty Records Excluded         : {empty_count:,}")
    print(f"3. Usable Dataset Records         : {usable_count:,}")
    print(f"4. Training Set (80%)             : {train_count:,} records")
    print(f"   - Fake News (0)                : {train_fake:,} ({train_fake / train_count * 100:.2f}%)")
    print(f"   - Real News (1)                : {train_real:,} ({train_real / train_count * 100:.2f}%)")
    print(f"5. Testing Set (20%)              : {test_count:,} records")
    print(f"   - Fake News (0)                : {test_fake:,} ({test_fake / test_count * 100:.2f}%)")
    print(f"   - Real News (1)                : {test_real:,} ({test_real / test_count * 100:.2f}%)")

    # ---------------------------------------------------------
    # 7. OVERLAP VERIFICATION
    # ---------------------------------------------------------
    print("\n[Step 6] Verifying data partition independence (Overlap Check)...")
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices.intersection(test_indices)

    if len(overlap) == 0:
        print(f"-> Overlap Verification PASSED: Exact 0 overlapping indices between train and test sets.")
    else:
        print(f"ERROR: Detected {len(overlap)} overlapping records between train and test sets!")
        return

    # ---------------------------------------------------------
    # 8. SAVE TRAIN AND TEST DATASETS
    # ---------------------------------------------------------
    print("\n[Step 7] Saving partitioned datasets to disk...")
    # Extract the full record information corresponding to train and test indices
    train_df = df_usable.loc[X_train.index].copy()
    test_df = df_usable.loc[X_test.index].copy()

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    train_size_mb = os.path.getsize(train_path) / (1024 * 1024)
    test_size_mb = os.path.getsize(test_path) / (1024 * 1024)

    print(f"-> Successfully saved Training dataset : '{os.path.basename(train_path)}' ({train_size_mb:.2f} MB, {len(train_df):,} rows)")
    print(f"-> Successfully saved Testing dataset  : '{os.path.basename(test_path)}' ({test_size_mb:.2f} MB, {len(test_df):,} rows)")
    print(f"-> Original datasets remain 100% untouched.")

    print("\n" + "=" * 80)
    print("PHASE 3 - STEP 1 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

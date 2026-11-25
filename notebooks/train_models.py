"""
Intelligent Fake News Detection Using NLP and Machine Learning
Phase 3 - Step 3: Machine Learning Model Training and Evaluation

This script:
1. Loads the precomputed sparse TF-IDF matrices (X_train_tfidf, X_test_tfidf) and labels (y_train, y_test).
2. Trains three prominent machine learning classifiers on the training data:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Passive Aggressive Classifier
3. Evaluates all three models on the unseen test data using standard evaluation metrics:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix (True Negatives, False Positives, False Negatives, True Positives)
   - Classification Report (Fake News vs. Real News)
4. Displays a formatted comparison table.
5. Saves all three trained models to 'models/' using joblib:
   - models/logistic_regression.pkl
   - models/naive_bayes.pkl
   - models/passive_aggressive.pkl
6. Saves comparison metrics to 'reports/model_comparison.csv'.
7. Validates persistence by re-loading all models from disk.
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib

# Suppress minor version-specific deprecation notices for clean terminal reporting
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model_name, model, X_test, y_test):
    """
    Evaluates a trained classifier on the test set and calculates key metrics:
    Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Classification Report.
    """
    start_eval = time.time()
    y_pred = model.predict(X_test)
    eval_time = time.time() - start_eval

    # Calculate overall metrics (weighted to account for any slight sample variations)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Binary class metrics specifically for Fake (0) and Real (1)
    prec_fake = precision_score(y_test, y_pred, pos_label=0)
    rec_fake = recall_score(y_test, y_pred, pos_label=0)
    f1_fake = f1_score(y_test, y_pred, pos_label=0)

    prec_real = precision_score(y_test, y_pred, pos_label=1)
    rec_real = recall_score(y_test, y_pred, pos_label=1)
    f1_real = f1_score(y_test, y_pred, pos_label=1)

    cm = confusion_matrix(y_test, y_pred)
    target_names = ['Fake News (0)', 'Real News (1)']
    clf_report = classification_report(y_test, y_pred, target_names=target_names, digits=4)

    return {
        'model_name': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'precision_fake': prec_fake,
        'recall_fake': rec_fake,
        'f1_fake': f1_fake,
        'precision_real': prec_real,
        'recall_real': rec_real,
        'f1_real': f1_real,
        'confusion_matrix': cm,
        'classification_report': clf_report,
        'eval_time': eval_time
    }


def print_model_evaluation(results):
    """Prints a clear, beginner-friendly evaluation summary for a model."""
    name = results['model_name']
    cm = results['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {name.upper()}")
    print("=" * 80)
    print(f"1. Overall Accuracy : {results['accuracy'] * 100:.2f}%")
    print(f"2. Precision        : {results['precision'] * 100:.2f}%")
    print(f"3. Recall           : {results['recall'] * 100:.2f}%")
    print(f"4. F1-Score         : {results['f1_score'] * 100:.2f}%")

    print("\n5. Confusion Matrix Breakdown:")
    print(f"   * True Negatives  (Actual Fake, Predicted Fake) : {tn:,}")
    print(f"   * False Positives (Actual Fake, Predicted Real) : {fp:,}")
    print(f"   * False Negatives (Actual Real, Predicted Fake) : {fn:,}")
    print(f"   * True Positives  (Actual Real, Predicted Real) : {tp:,}")

    print("\n   Visual Matrix:")
    print(f"                      Predicted Fake    Predicted Real")
    print(f"   Actual Fake (0) :  {tn:^14,}    {fp:^14,}")
    print(f"   Actual Real (1) :  {fn:^14,}    {tp:^14,}")

    print("\n6. Detailed Classification Report:")
    print(results['classification_report'])


def main():
    print("=" * 80)
    print("PHASE 3 - STEP 3: MACHINE LEARNING MODEL TRAINING & BENCHMARKING")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. SETUP PATHS
    # ---------------------------------------------------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")

    X_train_path = os.path.join(models_dir, "X_train_tfidf.npz")
    X_test_path = os.path.join(models_dir, "X_test_tfidf.npz")
    y_train_path = os.path.join(models_dir, "y_train.csv")
    y_test_path = os.path.join(models_dir, "y_test.csv")

    lr_save_path = os.path.join(models_dir, "logistic_regression_lr.pkl")
    nb_save_path = os.path.join(models_dir, "naive_bayes_lr.pkl")
    pa_save_path = os.path.join(models_dir, "passive_aggressive_lr.pkl")
    comparison_csv_path = os.path.join(reports_dir, "leakage_reduced_model_comparison.csv")

    os.makedirs(reports_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. LOAD PRECOMPUTED TF-IDF MATRICES & LABELS
    # ---------------------------------------------------------
    print("\n[Step 1] Loading precomputed sparse TF-IDF matrices & labels...")
    required_files = [X_train_path, X_test_path, y_train_path, y_test_path]
    for p in required_files:
        if not os.path.exists(p):
            print(f"ERROR: Required file not found: {p}")
            print("Please ensure Phase 3 Step 2 has completed successfully.")
            sys.exit(1)

    X_train = sp.load_npz(X_train_path)
    X_test = sp.load_npz(X_test_path)
    y_train = pd.read_csv(y_train_path)['label'].values
    y_test = pd.read_csv(y_test_path)['label'].values

    print(f"-> Training matrix loaded : {X_train.shape} ({X_train.nnz:,} non-zero features)")
    print(f"-> Testing matrix loaded  : {X_test.shape} ({X_test.nnz:,} non-zero features)")
    print(f"-> Training labels loaded : {len(y_train):,} samples (Fake: {(y_train == 0).sum():,}, Real: {(y_train == 1).sum():,})")
    print(f"-> Testing labels loaded  : {len(y_test):,} samples (Fake: {(y_test == 0).sum():,}, Real: {(y_test == 1).sum():,})")

    # ---------------------------------------------------------
    # 3. DEFINE MODELS TO TRAIN
    # ---------------------------------------------------------
    models_to_train = {
        "Logistic Regression": {
            "instance": LogisticRegression(max_iter=1000, random_state=42),
            "save_path": lr_save_path,
            "description": "Linear probabilistic classifier that models the log-odds of fake vs real news."
        },
        "Multinomial Naive Bayes": {
            "instance": MultinomialNB(),
            "save_path": nb_save_path,
            "description": "Probabilistic model based on Bayes' Theorem, ideal for discrete word count/TF-IDF distributions."
        },
        "Passive Aggressive Classifier": {
            "instance": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
            "save_path": pa_save_path,
            "description": "Online margin-based linear algorithm that updates aggressively upon classification error."
        }
    }

    all_results = []

    # ---------------------------------------------------------
    # 4. TRAIN AND EVALUATE EACH MODEL
    # ---------------------------------------------------------
    for model_name, info in models_to_train.items():
        print("\n" + "=" * 80)
        print(f"TRAINING: {model_name.upper()}")
        print("=" * 80)
        print(f"Description: {info['description']}")
        print(f"Training on X_train_tfidf ({X_train.shape[0]:,} records) and y_train...")

        model = info["instance"]
        start_train = time.time()
        # Train strictly on training data
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        print(f"-> Training completed in {train_time:.2f} seconds.")

        print(f"Evaluating on unseen X_test_tfidf ({X_test.shape[0]:,} records)...")
        results = evaluate_model(model_name, model, X_test, y_test)
        results['train_time'] = train_time
        results['model_instance'] = model
        results['save_path'] = info['save_path']

        print_model_evaluation(results)
        all_results.append(results)

        # Save trained model to disk
        print(f"Saving trained model to: {info['save_path']}")
        joblib.dump(model, info['save_path'])
        saved_size_kb = os.path.getsize(info['save_path']) / 1024
        print(f"-> Model saved successfully ({saved_size_kb:.2f} KB).")

    # ---------------------------------------------------------
    # 5. MODEL COMPARISON TABLE
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 80)

    comparison_data = []
    for r in all_results:
        comparison_data.append({
            "Model Name": r['model_name'],
            "Accuracy (%)": round(r['accuracy'] * 100, 2),
            "Precision (%)": round(r['precision'] * 100, 2),
            "Recall (%)": round(r['recall'] * 100, 2),
            "F1-Score (%)": round(r['f1_score'] * 100, 2),
            "Fake F1 (%)": round(r['f1_fake'] * 100, 2),
            "Real F1 (%)": round(r['f1_real'] * 100, 2),
            "Training Time (s)": round(r['train_time'], 3)
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Format terminal table
    print(f"{'Model Name':<32} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Train Time'}")
    print("-" * 92)
    for row in comparison_data:
        print(f"{row['Model Name']:<32} | {row['Accuracy (%)']:>8.2f}% | {row['Precision (%)']:>8.2f}% | {row['Recall (%)']:>8.2f}% | {row['F1-Score (%)']:>8.2f}% | {row['Training Time (s)']:>7.3f}s")

    # Save comparison metrics to reports/
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\n-> Comparison table saved to: {comparison_csv_path}")

    # ---------------------------------------------------------
    # 6. PERSISTENCE VALIDATION (RE-LOAD SAVED MODELS)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("VALIDATING SAVED MODEL PERSISTENCE (DISK RE-LOAD CHECK)")
    print("=" * 80)

    for r in all_results:
        path = r['save_path']
        loaded_model = joblib.load(path)
        # Verify predictions match exactly
        reloaded_acc = accuracy_score(y_test, loaded_model.predict(X_test))
        assert abs(reloaded_acc - r['accuracy']) < 1e-6, f"Accuracy mismatch on reload for {r['model_name']}"
        print(f"-> Verified: '{os.path.basename(path)}' re-loaded and validated (Accuracy: {reloaded_acc * 100:.2f}%). [PASSED]")

    print("\n" + "=" * 80)
    print("PHASE 3 - STEP 3 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

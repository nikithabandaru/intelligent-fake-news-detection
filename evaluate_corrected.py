import os, pandas as pd, pickle, textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
base = r'C:/Users/Nikitha/Downloads/Intelligent-Fake-News-Detection'
train_path = os.path.join(base, 'dataset/clean_split/train_clean.csv')
test_path = os.path.join(base, 'dataset/clean_split/test_clean_v2.csv')
model_dir = os.path.join(base, 'models')
report_dir = os.path.join(base, 'reports')

os.makedirs(model_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# ---------- Step 1: Data validation ----------
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print('--- Validation ---')
print(f'Train rows: {len(train)} (expected 31007)')
print(f'Test rows : {len(test)} (expected 7749)')

# Check missing normalized_text
missing_train = train['normalized_text'].isnull() | train['normalized_text'].str.strip().eq('')
missing_test = test['normalized_text'].isnull() | test['normalized_text'].str.strip().eq('')
print('Missing normalized_text - train:', missing_train.sum())
print('Missing normalized_text - test :', missing_test.sum())

# Check label values
print('Unique labels in train:', train['label'].unique())
print('Unique labels in test :', test['label'].unique())

# ---------- Step 2: Model training ----------
vectorizer = TfidfVectorizer(max_features=50000,
                             ngram_range=(1,2),
                             min_df=2,
                             max_df=0.95,
                             sublinear_tf=True)
X_train = vectorizer.fit_transform(train['normalized_text'])
y_train = train['label']

clf = LogisticRegression(max_iter=1000, n_jobs=5, solver='lbfgs')
clf.fit(X_train, y_train)

# ---------- Step 3: Evaluation ----------
X_test = vectorizer.transform(test['normalized_text'])
y_test = test['label']

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

# Macro metrics (since binary they equal micro but compute anyway)
macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

class_report = classification_report(y_test, y_pred, digits=4)
conf_mat = confusion_matrix(y_test, y_pred)

# Identify false positives and false negatives indices (positional)
fp_indices = test.index[(y_test == 0) & (y_pred == 1)].tolist()
fn_indices = test.index[(y_test == 1) & (y_pred == 0)].tolist()

# ---------- Step 4: Save artifacts ----------
vectorizer_path = os.path.join(model_dir, 'final_corrected_tfidf_vectorizer.pkl')
model_path = os.path.join(model_dir, 'final_corrected_logistic_regression.pkl')
report_path = os.path.join(report_dir, 'final_corrected_model_evaluation.txt')
conf_mat_path = os.path.join(report_dir, 'final_corrected_confusion_matrix.png')

# Ensure we do not overwrite existing files
for p in [vectorizer_path, model_path, report_path, conf_mat_path]:
    if os.path.exists(p):
        raise FileExistsError(f'File already exists and would be overwritten: {p}')

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

# Create evaluation report text
report_lines = []
report_lines.append('Logistic Regression evaluation on corrected test split')
report_lines.append('')
report_lines.append('--- Metrics ---')
report_lines.append(f'Accuracy            : {acc:.4f}')
report_lines.append(f'Precision (pos)    : {prec:.4f}')
report_lines.append(f'Recall (pos)       : {rec:.4f}')
report_lines.append(f'F1-score (pos)     : {f1:.4f}')
report_lines.append(f'Macro Precision    : {macro_prec:.4f}')
report_lines.append(f'Macro Recall       : {macro_rec:.4f}')
report_lines.append(f'Macro F1-score     : {macro_f1:.4f}')
report_lines.append('')
report_lines.append('--- Classification Report ---')
report_lines.append(class_report)
report_lines.append('')
report_lines.append('--- Confusion Matrix ---')
report_lines.append(str(conf_mat))
report_lines.append('')
report_lines.append(f'False Positive test indices (label=0 predicted=1): {fp_indices}')
report_lines.append(f'False Negative test indices (label=1 predicted=0): {fn_indices}')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns = sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(conf_mat_path)
plt.close()

# ---------- Step 5: Verification prints ----------
print('\n--- Verification Summary ---')
print(f'Model trained on train_clean.csv rows: {len(train)}')
print(f'Evaluation performed on test_clean_v2.csv rows: {len(test)}')
print('Vectorizer and model saved to:')
print(f'  {vectorizer_path}')
print(f'  {model_path}')
print('Report saved to:')
print(f'  {report_path}')
print('Confusion matrix image saved to:')
print(f'  {conf_mat_path}')

import os, pandas as pd, sys, textwrap

base = r'C:/Users/Nikitha/Downloads/Intelligent-Fake-News-Detection/dataset/clean_split'
train_path = os.path.join(base, 'train_clean.csv')
test_path = os.path.join(base, 'test_clean.csv')
leak_path = os.path.join(base, 'leak_report.csv')

# Load data
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)
leak = pd.read_csv(leak_path)

# Verify original test size
orig_test_len = len(test)
print(f'Original test rows: {orig_test_len}')

# Target indices
pairs = [
    {'train_pos':3174, 'test_pos':1075},
    {'train_pos':14457, 'test_pos':1626},
]

print('\n--- Displaying verified pairs ---')
for p in pairs:
    tp = p['train_pos']
    ts = p['test_pos']
    train_text = str(train.iloc[tp]['normalized_text'])
    test_text = str(test.iloc[ts]['normalized_text'])
    print(f"\nPair train_pos={tp}, test_pos={ts}")
    print('Train label:', train.iloc[tp]['label'])
    print('Test label :', test.iloc[ts]['label'])
    print('Train text length:', len(train_text))
    print('Test text length :', len(test_text))
    print('--- Train preview ---')
    print(textwrap.shorten(train_text, width=400, placeholder='...'))
    print('--- Test preview ---')
    print(textwrap.shorten(test_text, width=400, placeholder='...'))
    print('Containment check (train in test):', train_text in test_text)
    print('Containment check (test in train):', test_text in train_text)

# Remove rows from test using positional indices
remove_idxs = [p['test_pos'] for p in pairs]
# Ensure sorted descending to avoid index shift - pandas drop works with original index labels
test_v2 = test.drop(remove_idxs).reset_index(drop=True)
new_test_len = len(test_v2)
print(f'\nNew test rows after removal: {new_test_len}')

# Save new test file
test_v2_path = os.path.join(base, 'test_clean_v2.csv')
test_v2.to_csv(test_v2_path, index=False)
print('Saved new test file to', test_v2_path)

# Create removal report
def get_similarity(train_pos, test_pos):
    row = leak[(leak['train_pos']==train_pos) & (leak['test_pos']==test_pos)]
    if not row.empty:
        return float(row.iloc[0]['similarity'])
    return None

report_rows = []
for p in pairs:
    sim = get_similarity(p['train_pos'], p['test_pos'])
    report_rows.append({
        'removed_test_pos': p['test_pos'],
        'corresponding_train_pos': p['train_pos'],
        'similarity': sim,
        'train_label': train.iloc[p['train_pos']]['label'],
        'test_label': test.iloc[p['test_pos']]['label'],
        'removal_reason': 'containment leakage'
    })

report_df = pd.DataFrame(report_rows, columns=['removed_test_pos','corresponding_train_pos','similarity','train_label','test_label','removal_reason'])
report_path = os.path.join(base, 'removal_report_v2.csv')
report_df.to_csv(report_path, index=False)
print('Saved removal report to', report_path)

# Verification that only these rows were removed
# Re-load saved file to double‑check
test_v2_check = pd.read_csv(test_v2_path)
assert len(test_v2_check) == new_test_len, 'Row count mismatch after reload'
# Ensure all other rows are identical and in same order
mask = test.index.isin(remove_idxs)
expected = test[~mask].reset_index(drop=True)
if expected.equals(test_v2_check):
    print('Verification passed: only the targeted rows were removed and order preserved.')
else:
    print('Verification FAILED: discrepancy between expected and new test file.')

print('\n--- Summary ---')
print(f'Original test rows: {orig_test_len}')
print(f'Rows removed: {len(remove_idxs)}')
print(f'New test rows: {new_test_len}')
print('Removal report rows:', len(report_df))

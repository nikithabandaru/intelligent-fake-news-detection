import pandas as pd, json, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

project_root = r'C:\\Users\\Nikitha\\Downloads\\Intelligent-Fake-News-Detection'
train_path = os.path.join(project_root, 'dataset', 'leakage_reduced_train.csv')
test_path = os.path.join(project_root, 'dataset', 'leakage_reduced_test.csv')

# Load data
ttrain_df = pd.read_csv(train_path, usecols=['normalized_text', 'label'])
test_df = pd.read_csv(test_path, usecols=['normalized_text', 'label'])

# Vectorizer parameters matching original audit
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True, lowercase=False)
combined = pd.concat([ttrain_df['normalized_text'], test_df['normalized_text']])
vectorizer.fit(combined.astype(str))
X_train = vectorizer.transform(ttrain_df['normalized_text'].astype(str))
X_test = vectorizer.transform(test_df['normalized_text'].astype(str))

threshold = 0.90
nn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
nn.fit(X_train)
# radius = 1 - similarity threshold
distances, indices = nn.radius_neighbors(X_test, radius=1 - threshold, return_distance=True)

pairs = []
for test_idx, (dists, train_idxs) in enumerate(zip(distances, indices)):
    for dist, train_idx in zip(dists, train_idxs):
        sim = 1 - dist
        if sim >= threshold:
            pairs.append({
                'train_idx': int(train_idx),
                'test_idx': int(test_idx),
                'similarity': float(sim),
                'train_label': int(ttrain_df.iloc[train_idx]['label']),
                'test_label': int(test_df.iloc[test_idx]['label']),
                'train_text': ttrain_df.iloc[train_idx]['normalized_text'],
                'test_text': test_df.iloc[test_idx]['normalized_text']
            })

# Classification heuristic
categories = {
    'same_article_near_copy': [],
    'same_event_different_article': [],
    'different_stories_superficial': [],
    'uncertain': []
}
for p in pairs:
    txt1 = p['train_text']
    txt2 = p['test_text']
    sim = p['similarity']
    if txt1 == txt2:
        cat = 'same_article_near_copy'
        reason = 'Exact text match'
    elif sim >= 0.98:
        cat = 'same_article_near_copy'
        reason = f'High similarity ({sim:.3f}) suggests near‑copy'
    elif p['train_label'] == p['test_label']:
        cat = 'same_event_different_article'
        reason = 'Same label and moderate similarity'
    else:
        cat = 'different_stories_superficial'
        reason = 'Different labels; likely different news'
    p['category'] = cat
    p['reason'] = reason
    categories[cat].append(p)

summary = {k: len(v) for k, v in categories.items()}
output = {
    'total_pairs': len(pairs),
    'category_counts': summary,
    'pairs': pairs
}
print(json.dumps(output, indent=2, ensure_ascii=True))

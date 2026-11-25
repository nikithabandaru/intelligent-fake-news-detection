import os, json, pandas as pd, traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------- Configuration ----------
project_root = r'C:\\Users\\Nikitha\\Downloads\\Intelligent-Fake-News-Detection'
news_path = os.path.join(project_root, 'dataset', 'leakage_reduced_news.csv')
train_path = os.path.join(project_root, 'dataset', 'leakage_reduced_train.csv')
test_path = os.path.join(project_root, 'dataset', 'leakage_reduced_test.csv')

try:
    # ---------- Load data (read-only) ----------
    print('START: Loading full leakage_reduced_news...')
    news_df = pd.read_csv(news_path, usecols=['normalized_text', 'label'])
    print('DONE: Loaded full news, shape:', news_df.shape)
    print('START: Loading train split...')
    train_df = pd.read_csv(train_path, usecols=['normalized_text', 'label'])
    print('DONE: Loaded train split, shape:', train_df.shape)
    print('START: Loading test split...')
    test_df = pd.read_csv(test_path, usecols=['normalized_text', 'label'])
    print('DONE: Loaded test split, shape:', test_df.shape)

    # ---------- Exact duplicate analysis ----------
    print('START: Exact duplicate analysis')
    exact_counts = news_df['normalized_text'].value_counts()
    exact_dupes = exact_counts[exact_counts > 1]
    exact_duplicate_groups = int(len(exact_dupes))
    exact_duplicate_extra = int(exact_dupes.sum() - exact_duplicate_groups)
    print('DONE: Exact duplicate analysis -> groups:', exact_duplicate_groups,
          'extra rows:', exact_duplicate_extra)

    # ---------- Near-duplicate analysis (TF-IDF cosine similarity) ----------
    print('START: Vectorizing normalized_text for near-duplicate detection...')
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2),
                                 min_df=2, max_df=0.95,
                                 sublinear_tf=True, lowercase=False)
    X_all = vectorizer.fit_transform(news_df['normalized_text'].astype(str))
    print('DONE: Vectorization complete, matrix shape', X_all.shape)

    thresholds = [0.90, 0.95, 0.98]
    near_counts = {}
    near_examples = {}
    for thr in thresholds:
        print(f'START: Near-duplicate detection at {thr*100:.0f}%')
        radius = 1 - thr
        nn = NearestNeighbors(metric='cosine', radius=radius,
                              algorithm='brute', n_jobs=-1)
        nn.fit(X_all)
        distances, indices = nn.radius_neighbors(X_all, return_distance=True)
        pairs = set()
        for i, (dlist, ilist) in enumerate(zip(distances, indices)):
            for d, j in zip(dlist, ilist):
                if i >= j:
                    continue
                sim = 1 - d
                if sim >= thr:
                    pairs.add((i, j, sim))
        near_counts[thr] = len(pairs)
        # store up to 2 examples per threshold
        examples = []
        for i, j, sim in list(pairs)[:2]:
            examples.append({
                'idx1': int(i),
                'idx2': int(j),
                'similarity': sim,
                'label1': int(news_df.iloc[i]['label']),
                'label2': int(news_df.iloc[j]['label']),
                'text1': news_df.iloc[i]['normalized_text'][:200] + '...',
                'text2': news_df.iloc[j]['normalized_text'][:200] + '...'
            })
        near_examples[thr] = examples
        print(f'DONE: {thr*100:.0f}% -> {len(pairs)} pairs')

    # ---------- Cross-split duplicate / near-duplicate ----------
    print('START: Cross-split exact duplicate check')
    train_set = set(train_df['normalized_text'])
    test_set = set(test_df['normalized_text'])
    cross_exact = train_set.intersection(test_set)
    cross_exact_count = len(cross_exact)
    print('DONE: Cross-split exact duplicates count:', cross_exact_count)

    print('START: Cross-split near-duplicate detection')
    X_train = vectorizer.transform(train_df['normalized_text'].astype(str))
    X_test = vectorizer.transform(test_df['normalized_text'].astype(str))
    nn_cross = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    nn_cross.fit(X_train)
    cross_near_counts = {}
    cross_near_examples = {}
    for thr in thresholds:
        radius = 1 - thr
        distances, indices = nn_cross.radius_neighbors(X_test, radius=radius, return_distance=True)
        pairs = []
        for test_i, (dlist, train_idxs) in enumerate(zip(distances, indices)):
            for d, train_i in zip(dlist, train_idxs):
                sim = 1 - d
                if sim >= thr:
                    pairs.append((int(train_i), int(test_i), sim))
        cross_near_counts[thr] = len(pairs)
        examples = []
        for train_i, test_i, sim in pairs[:2]:
            examples.append({
                'train_idx': train_i,
                'test_idx': test_i,
                'similarity': sim,
                'train_label': int(train_df.iloc[train_i]['label']),
                'test_label': int(test_df.iloc[test_i]['label']),
                'train_text': train_df.iloc[train_i]['normalized_text'][:200] + '...',
                'test_text': test_df.iloc[test_i]['normalized_text'][:200] + '...'
            })
        cross_near_examples[thr] = examples
        print(f'DONE: Cross-split {thr*100:.0f}% -> {len(pairs)} pairs')

    # ---------- Label conflicts for exact duplicates ----------
    print('START: Label conflict detection')
    conflict_examples = []
    for txt, grp in news_df.groupby('normalized_text'):
        if grp['label'].nunique() > 1:
            conflict_examples.append({
                'text_snippet': txt[:200] + '...',
                'labels': sorted(grp['label'].unique().tolist()),
                'count': int(len(grp))
            })
    conflict_examples = conflict_examples[:5]
    print('DONE: Label conflict detection –', len(conflict_examples), 'examples')

    # ---------- Compile results ----------
    result = {
        'exact_duplicate_groups': exact_duplicate_groups,
        'exact_duplicate_extra_rows': exact_duplicate_extra,
        'near_duplicate_counts': {int(thr*100): cnt for thr, cnt in near_counts.items()},
        'near_duplicate_examples': {int(thr*100): ex for thr, ex in near_examples.items()},
        'cross_split_exact_duplicate_count': cross_exact_count,
        'cross_split_near_duplicate_counts': {int(thr*100): cnt for thr, cnt in cross_near_counts.items()},
        'cross_split_near_duplicate_examples': {int(thr*100): ex for thr, ex in cross_near_examples.items()},
        'label_conflict_examples': conflict_examples,
        'audit_columns_used': ['normalized_text', 'label']
    }
    print('--- Audit Results (JSON) ---')
    print(json.dumps(result, indent=2))
except Exception as e:
    print('ERROR: An exception occurred')
    traceback.print_exc()
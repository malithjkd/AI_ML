# Activity Recognition Model Training Guide

## 1. Purpose
Provide a structured path to train and compare multiple model families for classifying PAMAP2 activity IDs using pre-generated, cleaned 1‑second sensor chunks (100 samples @ 100 Hz, 40 features). Focus first on fast, information-rich baselines before moving to heavier deep sequence models.

## 2. Data Assumptions
- Chunk directory: `C:\Users\malit\Documents\GitHub\AI_ML_data\PAMAP2_Dataset\activity_chunks_v1`.
- File naming: `subject<id>_activity_<ACT>_chunk_<N>.csv` (all rows in one file = single activity segment).
- Each file: shape `(100, C_total)` after preprocessing (no NaN, zero-only columns removed).
- Column order consistent with preprocessing (`timestamp`, `activity_id`, `heart_rate`, ...sensor channels...).
- Target initial activity IDs: `[3,5,6,7]` (Standing, Running, Cycling, Nordic walking) — adjust as needed.
- Use exactly 50 randomly sampled chunks per class for first prototypes (balanced set).
- Exclude `timestamp` (no added value with uniform sampling) and `activity_id` (label only).

## 3. Model Roadmap & Rationale
| Stage | Model | Why Start / Use | Cost | When to Move On |
|-------|-------|-----------------|------|-----------------|
| 0 | Statistical Features + RandomForest | Fast sanity check; catches pipeline issues | Very Low | After confirming > chance accuracy |
| 1 | MiniRocket + RidgeClassifier (Primary First Experiment) | Strong time-series baseline; minimal tuning | Low | When exploring deeper architectures |
| 2 | 1D CNN | Learns local temporal patterns; GPU efficient | Low–Med | After MiniRocket baseline established |
| 3 | LSTM (or GRU) | Sequence memory on longer windows | Medium | After moving to >1s or overlapping windows |
| 4 | TCN / InceptionTime | SOTA conv architectures for multivariate signals | Medium–High | When seeking accuracy gains |
| 5 | Hybrid / Transformer | Advanced experimentation | High | Only if prior models plateau |

## 4. Baseline 0: Statistical Features + RandomForest
A pipeline transforming each 1‑second chunk into a fixed feature vector.

### Features Per Channel (per 1s window)
- mean, std, min, max, (max-min), energy (sum(x^2)/len)
Total features = (#channels_kept * 6)

### Purpose
- Validate dataset integrity
- Quick reference accuracy (e.g., expect well above random if signals are discriminative)

### Code Skeleton
```
import os, glob, re, random, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

DATA_DIR = r"C:\\Users\\malit\\Documents\\GitHub\\AI_ML_data\\PAMAP2_Dataset\\activity_chunks_v1"
TARGET = [3,5,6,7]
SAMPLES_PER_CLASS = 50
rng = random.Random(42)

def parse_activity(fp):
    m = re.search(r"activity_(\d+)_chunk", os.path.basename(fp))
    return int(m.group(1)) if m else None

# Collect
paths = [p for p in glob.glob(os.path.join(DATA_DIR,'*.csv')) if parse_activity(p) in TARGET]
by_act = {a: [] for a in TARGET}
for p in paths: by_act[parse_activity(p)].append(p)
for a in TARGET: rng.shuffle(by_act[a])
sel = []
for a in TARGET: sel.extend(by_act[a][:SAMPLES_PER_CLASS])

Xf, y = [], []
for fp in sel:
    arr = np.loadtxt(fp, delimiter=',', dtype=float)
    feat_idx = [i for i in range(arr.shape[1]) if i not in (0,1)]  # drop timestamp & activity
    seq = arr[:, feat_idx]  # (100, F)
    # compute features
    mu = seq.mean(axis=0)
    sd = seq.std(axis=0, ddof=0)
    mn = seq.min(axis=0)
    mx = seq.max(axis=0)
    rg = mx - mn
    energy = (seq**2).sum(axis=0)/seq.shape[0]
    feat_vec = np.concatenate([mu, sd, mn, mx, rg, energy])
    Xf.append(feat_vec)
    y.append(parse_activity(fp))

Xf = np.vstack(Xf)
y = np.array(y)

# Train/val/test split
idx = np.arange(len(Xf)); rng.shuffle(list(idx))
tr_end = int(0.7*len(idx)); va_end = int(0.85*len(idx))
tr, va, te = idx[:tr_end], idx[tr_end:va_end], idx[va_end:]
Xtr, Xva, Xte = Xf[tr], Xf[va], Xf[te]
ytr, yva, yte = y[tr], y[va], y[te]

scaler = StandardScaler().fit(Xtr)
Xtr = scaler.transform(Xtr); Xva = scaler.transform(Xva); Xte = scaler.transform(Xte)

clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
clf.fit(Xtr, ytr)
print('Val acc:', clf.score(Xva, yva))
print('Test acc:', clf.score(Xte, yte))
print(classification_report(yte, clf.predict(Xte)))
print(confusion_matrix(yte, clf.predict(Xte)))
```

### Interpret
If performance is very low (<60% accuracy on balanced 4-class), revisit preprocessing or feature list; otherwise proceed to MiniRocket.

## 5. Baseline 1 (Primary): MiniRocket + RidgeClassifier
Chosen as first serious benchmark due to speed and strong accuracy on short, multivariate sequences.

### Implementation Plan (Step-by-Step)
1. Enumerate files & group by activity ID.
2. Randomly sample 50 chunks per target class (seeded).
3. Load each chunk; drop `timestamp` and `activity_id` columns.
4. Keep first 40 sensor channels (or curated list) → shape `(100,40)`.
5. Transpose to `(40,100)` for MiniRocket (channels first).
6. Split dataset (70/15/15) with deterministic shuffle.
7. Fit `MiniRocketMultivariate` on training set only.
8. Transform train/val/test to feature matrices (large sparse-like dense float arrays).
9. Fit `RidgeClassifierCV` (log-spaced alphas) on transformed train; evaluate on val/test.
10. Save transformer, classifier, config (classes, channel indices).

### Full Code
```
import os, glob, re, random, numpy as np, joblib, json
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sktime.transformations.panel.minirocket import MiniRocketMultivariate

DATA_DIR = r"C:\\Users\\malit\\Documents\\GitHub\\AI_ML_data\\PAMAP2_Dataset\\activity_chunks_v1"
TARGET_ACTIVITIES = [3,5,6,7]
SAMPLES_PER_CLASS = 50
FEATURE_LIMIT = 40
SEQ_LEN = 100
SEED = 42
out_dir = 'models/minirocket_baseline'
os.makedirs(out_dir, exist_ok=True)

rng = random.Random(SEED)

def parse_activity(fp):
    m = re.search(r"activity_(\d+)_chunk", os.path.basename(fp))
    return int(m.group(1)) if m else None

# 1 Collect & group
paths = [p for p in glob.glob(os.path.join(DATA_DIR,'*.csv')) if parse_activity(p) in TARGET_ACTIVITIES]
by_act = {a: [] for a in TARGET_ACTIVITIES}
for p in paths: by_act[parse_activity(p)].append(p)
for a in TARGET_ACTIVITIES: rng.shuffle(by_act[a])

# 2 Sample balanced
selected = []
for a in TARGET_ACTIVITIES: selected.extend(by_act[a][:SAMPLES_PER_CLASS])

# 3-5 Load & build tensor list
X_list, y_list = [], []
for fp in selected:
    arr = np.loadtxt(fp, delimiter=',', dtype=float)
    feat_idx = [i for i in range(arr.shape[1]) if i not in (0,1)][:FEATURE_LIMIT]
    seq = arr[:, feat_idx].T   # (F, T)
    if seq.shape != (FEATURE_LIMIT, SEQ_LEN):
        continue
    X_list.append(seq)
    y_list.append(parse_activity(fp))

X = np.stack(X_list)  # (N, F, T)
y = np.array(y_list)

# 6 Split
idx = np.arange(len(X)); rng.shuffle(list(idx))
tr_end = int(0.7*len(idx)); va_end = int(0.85*len(idx))
tr, va, te = idx[:tr_end], idx[tr_end:va_end], idx[va_end:]
Xtr, Xva, Xte = X[tr], X[va], X[te]
ytr, yva, yte = y[tr], y[va], y[te]

# 7-8 MiniRocket fit/transform
mr = MiniRocketMultivariate(random_state=SEED)
mr.fit(Xtr)
Phi_tr = mr.transform(Xtr)
Phi_va = mr.transform(Xva)
Phi_te = mr.transform(Xte)

# 9 RidgeClassifierCV
clf = RidgeClassifierCV(alphas=np.logspace(-3,3,7))
clf.fit(Phi_tr, ytr)
print('Val acc:', clf.score(Phi_va, yva))
print('Test acc:', clf.score(Phi_te, yte))
print(classification_report(yte, clf.predict(Phi_te)))
print(confusion_matrix(yte, clf.predict(Phi_te)))

# 10 Save artifacts
joblib.dump(mr, os.path.join(out_dir,'transform.pkl'))
joblib.dump(clf, os.path.join(out_dir,'ridge.pkl'))
with open(os.path.join(out_dir,'config.json'),'w') as f:
    json.dump({'activities': TARGET_ACTIVITIES, 'feature_limit': FEATURE_LIMIT, 'seq_len': SEQ_LEN, 'seed': SEED}, f, indent=2)
```

### Metric Focus
- Accuracy (balanced set)
- Confusion matrix (misclassification pattern)
- Later: Macro F1 when class sizes diverge.

### Expected Outcome
MiniRocket typically yields a strong baseline quickly; use as comparison anchor for all subsequent models.

## 6. Baseline 2: 1D CNN (Planned)
Key design:
- Input shape `(B, T, F)` → transpose to `(B, F, T)`.
- 3 Conv blocks: (Conv1d → BN → ReLU → Dropout) with kernel sizes e.g. `[5,5,3]`, channels `[64,128,128]`.
- Global Average Pool over time → Dense → Softmax.
Advantages: speed, good with short windows. Add later when comparing representational approaches.

## 7. Baseline 3: LSTM (Deferred Summary)
Still valuable for longer context windows. Retain earlier in-depth section (can keep code in separate script). Core spec:
- 2-layer LSTM, hidden 128, dropout 0.2.
- Last-step representation → BatchNorm → Linear.
- Use only after establishing MiniRocket & CNN performance or when extending to 5.12 s windows.

(Full prior LSTM code kept in version control; move to `scripts/train_lstm.py`).

## 8. Baseline 4: TCN / InceptionTime (Future)
- TCN: Dilated causal convolutions capture multi-scale temporal structure efficiently.
- InceptionTime: Ensemble of inception modules with varied kernel lengths — strong for multivariate HAR tasks.
Add once simpler models plateau.

## 9. Evaluation Protocols
| Protocol | Description | When |
|----------|-------------|------|
| Random Chunk Split | Current 70/15/15 within one subject | Prototype
| Subject Stratified | Mixed subjects, same distribution | After adding more subjects
| Leave-One-Subject-Out | Train on N-1 subjects, test on 1 | Generalization study
| Sliding Window (5.12 s, 50% overlap) | Larger temporal context | Phase 2

## 10. Artifact Management
```
models/
  rf_baseline/
  minirocket_baseline/
  cnn_baseline/
  lstm_baseline/
```
Each folder: `config.json`, model weights (`.pkl` or `.pt`), preprocessing objects (scaler, transformer), metrics summary.

## 11. Reproducibility
- Fixed seed for: Python `random`, NumPy, PyTorch (when used).
- Log: versions of `numpy`, `scikit-learn`, `sktime`, `torch`.
- Store commit hash in `config.json` if using git hooks.

## 12. Transition to Longer Windows (5.12 s)
Steps:
1. Regenerate chunks with sliding window length 512 samples (5.12 s) & stride 100 (1 s shift) to mirror literature.
2. Re-run Baseline 0 (features will aggregate over more samples: potentially higher class separation).
3. Re-run MiniRocket (benefits from added temporal complexity).
4. Introduce LSTM / TCN (now able to exploit longer dependencies).

## 13. Troubleshooting
| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| All baselines low | Wrong feature indices / label parsing | Print sample rows & verify
| MiniRocket very slow | Large N or high channel count | Downsample channels or sample fewer chunks
| Overfitting CNN/LSTM | Too few windows | Data augmentation / longer windows
| Class confusion pairs | Physiologically similar activities | Add frequency features, longer context

## 14. Comparison Checklist (Update After Each Run)
| Model | Train Time | Val Acc | Test Acc | Notes |
|-------|-----------|---------|----------|-------|
| RandomForest | | | | |
| MiniRocket | | | | |
| 1D CNN | | | | |
| LSTM | | | | |
| TCN / InceptionTime | | | | |

## 15. Next Steps Prioritization
1. Run Baseline 0 & 1; log metrics.
2. Implement 1D CNN script; compare vs MiniRocket.
3. Decide if extending to longer windows earlier improves all models.
4. Add more subjects; move to subject-independent evaluation.
5. Introduce hyperparameter sweeps for best two architectures only.

## 16. Summary
The training flow now begins with fast, information-dense baselines (statistical features + RandomForest, MiniRocket) before investing in deep architectures. This reduces iteration time, provides robust performance anchors, and ensures later complexity (CNN, LSTM, TCN) yields measurable and justified gains.

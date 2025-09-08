# LSTM Activity Recognition Model Training Guide

## 1. Objective
Train an initial sequence model (LSTM) to classify physical activity IDs from pre-created 1‑second sensor chunks (100 time steps at 100 Hz) using 40 cleaned sensor features (timestamp excluded). This is a baseline to validate pipeline correctness before scaling to longer windows (e.g. 5.12 s) or subject‑independent evaluation.

## 2. Current Assumptions
- Data chunks already created: `C:\Users\malit\Documents\GitHub\AI_ML_data\PAMAP2_Dataset\activity_chunks_v1`.
- File pattern: `subject<id>_activity_<ACT>_chunk_<N>.csv` (all rows of a file belong to a single activity).
- Each CSV has exactly 100 consecutive rows (1 second) after cleaning (no NaN / all‑zero columns removed).
- Features per row: `[timestamp, activity_id, heart_rate, ...  (sensor signals) ]` — you will select 40 sensor channels (heart_rate optional) and drop: `timestamp`, `activity_id` (used only as label).
- Balanced prototype training: sample exactly 50 chunks per target class (e.g. activities 3, 5, 6, 7). No class weighting needed.
- GPU with CUDA available.

## 3. Recommended Framework
Use **PyTorch** for flexibility with sequence models and rapid experimentation on GPU.

Alternatives to try later:
- 1D CNN (faster, good baseline)
- GRU (lighter than LSTM)
- TCN (Temporal Convolutional Network)
- Simple statistical feature aggregation + RandomForest (sanity check)

## 4. Directory Layout (suggested additions)
```
Physical_activity_monitoring/
  models/
    lstm_baseline/
      model.pt
      scaler.pkl
      config.json
  scripts/
    train_lstm.py
    infer_lstm.py
```

## 5. Feature Selection Strategy
1. Load one sample chunk to inspect column order (reuse the `column_names` list from preprocessing stage, AFTER removal of static columns). Store that final list in a small Python module (e.g. `column_schema.py`) to avoid drift.
2. Exclude: `timestamp`, `activity_id`.
3. Keep 40 columns (heart_rate + 39 sensor signals). If >40 available, choose those with variance > 0 and domain relevance (e.g., all accelerometer (both ranges), gyroscope, magnetometer, orientation (if valid) temps, heart_rate). Save final list in config.

## 6. Data Loading & Sampling
Pseudo logic:
```
- Scan directory for files matching pattern *_activity_*.csv
- Parse activity ID from filename (split by '_' and take index after 'activity')
- Filter to target activity IDs (e.g. {3,5,6,7})
- Group file paths by activity ID
- For each activity ID: randomly select up to 50 files (set seed for reproducibility)
- For each file:
    read np.loadtxt(..., delimiter=',')  # shape (100, total_cols_after_cleaning)
    X_seq = rows[:, selected_feature_indices]
    y_label = activity_id
    Append
- Stack: X -> shape (N_samples, 100, 40); y -> shape (N_samples,)
```

## 7. Train / Val / Test Split
For single‑subject prototype:
- Shuffle dataset (fixed random seed) then split 70% / 15% / 15%.
- (Later) For multi‑subject: perform subject‑wise split or leave‑one‑subject‑out.

## 8. Scaling
Because temporal order matters, apply scaling per feature across the training set only:
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_2d = X_train.reshape(-1, num_features)  # (N_train*100, 40)
scaler.fit(X_train_2d)
X_train = scaler.transform(X_train_2d).reshape(N_train, seq_len, num_features)
Apply transform to val/test similarly.
```
Persist `scaler.pkl` with joblib.

## 9. PyTorch Dataset & DataLoader
```
class ChunkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
```

## 10. Model Architecture (Baseline LSTM)
```
class LSTMActivityNet(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, dropout=0.2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):      # x: (B, T, F)
        out, (h, c) = self.lstm(x)
        last = out[:, -1, :]   # (B, hidden)
        last = self.bn(last)
        logits = self.fc(last)
        return logits
```
Hyperparameters to start:
- hidden_size: 128
- num_layers: 2
- dropout: 0.2
- optimizer: Adam (lr=1e-3)
- epochs: 30 (early stop on val loss patience=5)
- loss: CrossEntropyLoss

## 11. Training Loop (Outline)
```
for epoch in range(epochs):
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    # validation
    model.eval(); correct=0; total=0; val_loss=0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            val_loss += criterion(logits, yb).item()*len(Xb)
            preds = logits.argmax(1)
            correct += (preds==yb).sum().item(); total += len(Xb)
    val_acc = correct/total; val_loss/=total
    # early stopping logic
```

## 12. Metrics & Reporting
- Primary: Accuracy (balanced classes).
- Additional: Per-class precision/recall, confusion matrix (sklearn).
- Log: epoch, train_loss, val_loss, val_acc.

## 13. Saving Artifacts
```
os.makedirs(out_dir, exist_ok=True)
torch.save(model.state_dict(), out_dir/"model.pt")
joblib.dump(scaler, out_dir/"scaler.pkl")
json.dump(config_dict, open(out_dir/"config.json","w"), indent=2)
```
Include: feature_names, target_activity_ids, class_to_index mapping, model hyperparams, random seeds.

## 14. Inference Pipeline
```
# load scaler, model
# load chunk csv -> array (100, total_cols)
# derive features -> (1, 100, 40)
# scale -> reshape -> model -> softmax
probs = torch.softmax(logits, dim=1)
pred_class = idx_to_class[int(probs.argmax())]
```
For streaming / longer windows: maintain rolling buffer of last 100 samples and re-run.

## 15. Timestamp Usage
Exclude absolute timestamps (model learns from sensor dynamics). If irregular spacing later matters, you can add a derived feature: Delta_t_mean (should be constant ~0.01s) — currently no benefit.

## 16. Extending the Baseline
| Area | Next Step |
|------|-----------|
| Longer Context | Use 5.12 s (512 samples) windows with 50% overlap. |
| Data Efficiency | Precompute z-score per subject; test subject-independent generalization. |
| Regularization | Add dropout between LSTM and FC; try weight decay=1e-4. |
| Alternative Models | 1D CNN (e.g., 3 conv blocks) often matches LSTM with lower latency. |
| Fusion | Compute magnitude features (acc_mag, gyro_mag) as extra channels. |
| Augmentation | Add Gaussian noise (std 0.01), small scaling of acceleration. |
| Evaluation | Add F1-score macro for imbalanced future scenarios. |

## 17. Example Minimal `train_lstm.py` Skeleton
```
import os, glob, re, json, random, joblib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

ACTIVITY_IDS = [3,5,6,7]
SAMPLES_PER_CLASS = 50
SEQ_LEN = 100
INPUT_SIZE = 40
DATA_DIR = r"C:\\Users\\malit\\Documents\\GitHub\\AI_ML_data\\PAMAP2_Dataset\\activity_chunks_v1"

def parse_activity(fp):
    m = re.search(r"activity_(\d+)_chunk", os.path.basename(fp))
    return int(m.group(1)) if m else None

# 1. Collect files
paths = [p for p in glob.glob(os.path.join(DATA_DIR,"*.csv")) if parse_activity(p) in ACTIVITY_IDS]
# 2. Group
by_act = {a: [] for a in ACTIVITY_IDS}
for p in paths: by_act[parse_activity(p)].append(p)
# 3. Sample
rng = random.Random(42)
for a in ACTIVITY_IDS: rng.shuffle(by_act[a])
selected = []
for a in ACTIVITY_IDS: selected.extend(by_act[a][:SAMPLES_PER_CLASS])
# 4. Load
X_list, y_list = [], []
for fp in selected:
    arr = np.loadtxt(fp, delimiter=',', dtype=float)
    if arr.shape[0] != SEQ_LEN: continue
    # define feature indices (example placeholder: drop col 0 timestamp, col 1 activity)
    feature_idxs = [i for i in range(arr.shape[1]) if i not in (0,1)][:INPUT_SIZE]
    X_list.append(arr[:, feature_idxs])
    y_list.append(parse_activity(fp))
X = np.stack(X_list)  # (N, T, F)
y = np.array(y_list)
# 5. Map labels -> indices
classes = sorted(ACTIVITY_IDS)
class_to_idx = {c:i for i,c in enumerate(classes)}
y_idx = np.array([class_to_idx[c] for c in y])
# 6. Split
idxs = np.arange(len(X)); rng.shuffle(list(idxs))
train_end = int(0.7*len(X)); val_end = int(0.85*len(X))
train_idx, val_idx, test_idx = idxs[:train_end], idxs[train_end:val_end], idxs[val_end:]
X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y_idx[train_idx], y_idx[val_idx], y_idx[test_idx]
# 7. Scale
scaler = StandardScaler().fit(X_train.reshape(-1, INPUT_SIZE))
X_train = scaler.transform(X_train.reshape(-1, INPUT_SIZE)).reshape(-1, SEQ_LEN, INPUT_SIZE)
X_val   = scaler.transform(X_val.reshape(-1, INPUT_SIZE)).reshape(-1, SEQ_LEN, INPUT_SIZE)
X_test  = scaler.transform(X_test.reshape(-1, INPUT_SIZE)).reshape(-1, SEQ_LEN, INPUT_SIZE)

class ChunkDS(Dataset):
    def __init__(self,X,y): self.X=torch.from_numpy(X).float(); self.y=torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i]

train_loader = DataLoader(ChunkDS(X_train,y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(ChunkDS(X_val,y_val), batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, classes=len(classes), dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers>1 else 0)
        self.bn = nn.BatchNorm1d(hidden)
        self.fc = nn.Linear(hidden, classes)
    def forward(self,x):
        out,_ = self.lstm(x)
        h_last = out[:,-1,:]
        return self.fc(self.bn(h_last))

model = LSTMNet(INPUT_SIZE).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

best_val = 0; patience=5; wait=0
for epoch in range(30):
    model.train(); total_loss=0
    for xb,yb in train_loader:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad(); logits = model(xb); loss=crit(logits,yb); loss.backward();
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); total_loss+=loss.item()*len(xb)
    # val
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb,yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(1)
            correct += (preds==yb).sum().item(); total += len(xb)
    val_acc = correct/total
    print(f"Epoch {epoch+1}: train_loss={total_loss/len(X_train):.4f} val_acc={val_acc:.3f}")
    if val_acc>best_val: best_val=val_acc; wait=0; torch.save(model.state_dict(), 'model.pt')
    else:
        wait+=1
        if wait>=patience: break

# Test
model.load_state_dict(torch.load('model.pt'))
model.eval();
X_test_t = torch.from_numpy(X_test).float().to(device)
with torch.no_grad(): preds = model(X_test_t).argmax(1).cpu().numpy()
print(classification_report(y_test, preds, target_names=[str(c) for c in classes]))
print(confusion_matrix(y_test, preds))

# Save scaler & config
import joblib, json
joblib.dump(scaler, 'scaler.pkl')
json.dump({'classes': classes, 'feature_count': INPUT_SIZE, 'seq_len': SEQ_LEN}, open('config.json','w'), indent=2)
```

## 18. Validation Checks
Before accepting results:
- Confirm each loaded chunk has uniform time deltas (already enforced earlier).
- Print per-class counts after sampling.
- Verify scaler not fit on val/test.
- Ensure no leakage: file list shuffle reproducible (fixed seed).

## 19. Next Steps After Baseline
1. Increase window length (5.12 s) + overlap (50%) → more temporal context.
2. Add second head predicting intensity groups (multi-task learning).
3. Add model export to TorchScript for deployment.
4. Introduce subject-independent evaluation (leave-one-subject-out loop).
5. Hyperparameter search (hidden sizes, layers, LR schedule, optimizer variants like Ranger/AdamW).

## 20. Troubleshooting
| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| Low val accuracy | Overfitting small sample; too short window | Increase window length / add dropout |
| Nan loss | Bad scaler fit or malformed chunk | Recompute scaler; skip irregular chunks |
| Class confusion | Signals too similar in 1s window | Use longer context or frequency features |
| GPU underutilized | Small batch or fast model | Increase batch size (64 / 128) |

## 21. Summary
This guide establishes a reproducible baseline LSTM classifier over 1‑second cleaned sensor sequences (40 features). It emphasizes strict separation of scaling, balanced class sampling, and modular artifact saving to enable rapid iteration toward more advanced temporal architectures.

## 22. MiniRocket Baseline (sktime)
A very strong, fast baseline using random convolutional kernels + linear classifier; useful to benchmark deep models.

### Why MiniRocket
- Trains in seconds on CPU for moderate dataset sizes.
- Often matches or exceeds LSTM / CNN on short fixed-length windows.
- No heavy hyperparameter tuning (only number of features).

### Installation
(Optional virtual env first.)
```
pip install sktime==0.28.0
```

### Data Preparation
MiniRocket expects shape: (n_instances, n_channels, series_length). Your current NumPy array: (N, T=100, F=40).
Transpose to (N, F, 100).

### Quick Script Snippet
```
import os, glob, re, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sktime.transformations.panel.minirocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV

DATA_DIR = r"C:\\Users\\malit\\Documents\\GitHub\\AI_ML_data\\PAMAP2_Dataset\\activity_chunks_v1"
TARGET_ACTIVITIES = [3,5,6,7]
SAMPLES_PER_CLASS = 50
SEQ_LEN = 100
FEATURE_LIMIT = 40  # keep consistency with LSTM baseline

import random, joblib
rng = random.Random(42)

def parse_activity(fp):
    m = re.search(r"activity_(\d+)_chunk", os.path.basename(fp))
    return int(m.group(1)) if m else None

# Collect & sample
paths = [p for p in glob.glob(os.path.join(DATA_DIR,'*.csv')) if parse_activity(p) in TARGET_ACTIVITIES]
by_act = {a: [] for a in TARGET_ACTIVITIES}
for p in paths: by_act[parse_activity(p)].append(p)
for a in TARGET_ACTIVITIES: rng.shuffle(by_act[a])
selected = []
for a in TARGET_ACTIVITIES: selected.extend(by_act[a][:SAMPLES_PER_CLASS])

X_list, y_list = [], []
for fp in selected:
    arr = np.loadtxt(fp, delimiter=',', dtype=float)
    # drop timestamp (0) & activity id (1), take first FEATURE_LIMIT columns after
    feat_idx = [i for i in range(arr.shape[1]) if i not in (0,1)][:FEATURE_LIMIT]
    seq = arr[:, feat_idx]          # (T, F)
    X_list.append(seq.T)            # (F, T)
    y_list.append(parse_activity(fp))

X = np.stack(X_list)  # (N, F, T)
y = np.array(y_list)

# Train/val/test split 70/15/15
idx = np.arange(len(X)); rng.shuffle(list(idx))
train_end = int(0.7*len(X)); val_end = int(0.85*len(X))
tr, va, te = idx[:train_end], idx[train_end:val_end], idx[val_end:]
X_train, X_val, X_test = X[tr], X[va], X[te]
y_train, y_val, y_test = y[tr], y[va], y[te]

# MiniRocket feature transform
mr = MiniRocketMultivariate(random_state=42)
mr.fit(X_train)
Phi_train = mr.transform(X_train)
Phi_val   = mr.transform(X_val)
Phi_test  = mr.transform(X_test)

clf = RidgeClassifierCV(alphas=np.logspace(-3,3,7))
clf.fit(Phi_train, y_train)

print('Val accuracy:', clf.score(Phi_val, y_val))
print('Test accuracy:', clf.score(Phi_test, y_test))
print(classification_report(y_test, clf.predict(Phi_test)))
print(confusion_matrix(y_test, clf.predict(Phi_test)))

# Save artifacts
os.makedirs('models/minirocket_baseline', exist_ok=True)
joblib.dump(mr, 'models/minirocket_baseline/minirocket_transform.pkl')
joblib.dump(clf, 'models/minirocket_baseline/ridge_classifier.pkl')
```

### Interpretation
- If MiniRocket outperforms LSTM notably, prioritize improving sequence window length or trying CNN/TCN before tuning LSTM.
- If LSTM is close, longer context (5.12 s) may shift advantage.

### When to Prefer MiniRocket
- Rapid experimentation.
- Limited GPU resources.
- Establishing a high baseline before deep model complexity.

## 23. Baseline Comparison Checklist
| Aspect | LSTM | MiniRocket |
|--------|------|------------|
| Training speed | Slower (GPU) | Very fast (CPU) |
| Feature engineering | End-to-end | Implicit kernels |
| Interpretability | Medium | Low (random conv kernels) |
| Scalability to longer windows | Good | Feature matrix grows; still efficient |

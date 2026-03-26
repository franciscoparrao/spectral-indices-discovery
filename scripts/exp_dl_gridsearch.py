#!/usr/bin/env python3
"""
E-M4: MLP grid search to demonstrate DL was reasonably tuned.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

RESULTS_DIR = Path("data/results")
GT_DIR = Path("data/ground_truth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(GT_DIR / "maricunga_training_s2_gee.npz", allow_pickle=True)
X_raw = data['X']
y = data['y']
band_names = list(data['band_names'])
B = {name: i for i, name in enumerate(band_names)}

eps = 1e-6
class_ids = sorted(np.unique(y))
n_classes = len(class_ids)
label_map = {c: i for i, c in enumerate(class_ids)}
y_mapped = np.array([label_map[v] for v in y])

# SR + classical features (best feature set)
sr_classical = np.column_stack([
    X_raw[:, B['B04']] - 0.135,
    0.83 - X_raw[:, B['B02']] / np.maximum(X_raw[:, B['B05']], eps),
    0.09 / np.maximum(X_raw[:, B['B05']], eps),
    X_raw[:, B['B03']] - 0.48 * X_raw[:, B['B11']],
    (np.sqrt(X_raw[:, B['B12']]) - X_raw[:, B['B11']]) ** 2,
    X_raw[:, B['B03']] * X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B07']]**2, eps) - 0.45,
    X_raw[:, B['B11']] / np.maximum(X_raw[:, B['B12']], eps),
    X_raw[:, B['B04']] / np.maximum(X_raw[:, B['B02']], eps),
    X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B8A']], eps),
    (X_raw[:, B['B11']] - X_raw[:, B['B12']]) / np.maximum(X_raw[:, B['B11']] + X_raw[:, B['B12']], eps),
    X_raw[:, B['B02']] / np.maximum(X_raw[:, B['B11']], eps),
    X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B11']], eps),
    (X_raw[:, B['B8A']] - X_raw[:, B['B04']]) / np.maximum(X_raw[:, B['B8A']] + X_raw[:, B['B04']], eps),
])

n_input = sr_classical.shape[1]


def compute_class_weights(y_train, n_classes):
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts[counts == 0] = 1
    weights = len(y_train) / (n_classes * counts)
    return torch.FloatTensor(weights).to(device)


def train_eval_mlp(X_train, y_train, X_test, y_test, hidden, lr, epochs, batch_size=512):
    scaler = StandardScaler()
    X_tr = torch.FloatTensor(scaler.fit_transform(X_train)).to(device)
    X_te = torch.FloatTensor(scaler.transform(X_test)).to(device)
    y_tr = torch.LongTensor(y_train).to(device)

    model = nn.Sequential(
        nn.Linear(n_input, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, n_classes),
    ).to(device)

    weights = compute_class_weights(y_train, n_classes)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(X_tr, y_tr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in dl:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    return proba, preds


# Grid search
configs = []
for hidden in [32, 64, 128]:
    for lr in [1e-2, 1e-3, 1e-4]:
        for epochs in [30, 60, 100]:
            configs.append({'hidden': hidden, 'lr': lr, 'epochs': epochs})

print(f"Grid search: {len(configs)} configurations")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for i, cfg in enumerate(configs):
    fold_aucs = []
    fold_ba = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(sr_classical, y_mapped)):
        proba, preds = train_eval_mlp(
            sr_classical[train_idx], y_mapped[train_idx],
            sr_classical[test_idx], y_mapped[test_idx],
            cfg['hidden'], cfg['lr'], cfg['epochs']
        )

        fold_ba.append(balanced_accuracy_score(y_mapped[test_idx], preds))

        class_aucs = []
        for ci in range(n_classes):
            y_bin = (y_mapped[test_idx] == ci).astype(int)
            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                class_aucs.append(roc_auc_score(y_bin, proba[:, ci]))
        fold_aucs.append(np.mean(class_aucs))

    mean_auc = np.mean(fold_aucs)
    mean_ba = np.mean(fold_ba)

    result = {**cfg, 'mean_auc': float(mean_auc), 'ba': float(mean_ba),
              'auc_std': float(np.std(fold_aucs))}
    results.append(result)

    print(f"  [{i+1}/{len(configs)}] h={cfg['hidden']}, lr={cfg['lr']}, ep={cfg['epochs']}: "
          f"mAUC={mean_auc:.3f} BA={mean_ba:.3f}")

# Best config
best = max(results, key=lambda x: x['mean_auc'])
print(f"\nBest MLP config: hidden={best['hidden']}, lr={best['lr']}, "
      f"epochs={best['epochs']} → mAUC={best['mean_auc']:.3f}, BA={best['ba']:.3f}")

print(f"\nFor comparison: RF(SR+classical) = 0.920, SVM-RBF(SR+classical) = 0.926")

output = RESULTS_DIR / "dl_gridsearch.json"
with open(output, 'w') as f:
    json.dump({'configs': results, 'best': best}, f, indent=2)
print(f"Saved: {output}")

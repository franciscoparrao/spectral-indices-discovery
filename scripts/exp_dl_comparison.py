#!/usr/bin/env python3
"""
DL comparison: 1D-CNN and MLP on raw bands vs SR features.
Lightweight architectures — the point is to show that DL doesn't
substantially outperform RF(SR+classical).
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
print(f"Device: {device}")

# Load data
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

# SR features
sr_features = np.column_stack([
    X_raw[:, B['B04']] - 0.135,
    0.83 - X_raw[:, B['B02']] / np.maximum(X_raw[:, B['B05']], eps),
    0.09 / np.maximum(X_raw[:, B['B05']], eps),
    X_raw[:, B['B03']] - 0.48 * X_raw[:, B['B11']],
    (np.sqrt(X_raw[:, B['B12']]) - X_raw[:, B['B11']]) ** 2,
    X_raw[:, B['B03']] * X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B07']]**2, eps) - 0.45,
])

# Classical
classical_features = np.column_stack([
    X_raw[:, B['B11']] / np.maximum(X_raw[:, B['B12']], eps),
    X_raw[:, B['B04']] / np.maximum(X_raw[:, B['B02']], eps),
    X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B8A']], eps),
    (X_raw[:, B['B11']] - X_raw[:, B['B12']]) / np.maximum(X_raw[:, B['B11']] + X_raw[:, B['B12']], eps),
    X_raw[:, B['B02']] / np.maximum(X_raw[:, B['B11']], eps),
    X_raw[:, B['B12']] / np.maximum(X_raw[:, B['B11']], eps),
    (X_raw[:, B['B8A']] - X_raw[:, B['B04']]) / np.maximum(X_raw[:, B['B8A']] + X_raw[:, B['B04']], eps),
])

sr_classical = np.column_stack([sr_features, classical_features])


# ===== Model definitions =====

class MLP(nn.Module):
    """Simple MLP baseline."""
    def __init__(self, n_input, n_classes, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class Conv1DNet(nn.Module):
    """1D-CNN treating spectral bands as a 1D signal."""
    def __init__(self, n_input, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x: (batch, n_bands) → (batch, 1, n_bands)
        x = x.unsqueeze(1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


def compute_class_weights(y_train, n_classes):
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts[counts == 0] = 1
    weights = len(y_train) / (n_classes * counts)
    return torch.FloatTensor(weights).to(device)


def train_and_eval(model_class, X_train, y_train, X_test, y_test,
                   n_input, n_classes, epochs=50, batch_size=256, lr=1e-3):
    """Train a PyTorch model and return predictions."""
    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # To tensors
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.LongTensor(y_train).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)

    ds = TensorDataset(X_tr_t, y_tr_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = model_class(n_input, n_classes).to(device)
    weights = compute_class_weights(y_train, n_classes)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    for epoch in range(epochs):
        for xb, yb in dl:
            optimizer.zero_grad()
            out = criterion(model(xb), yb)
            out.backward()
            optimizer.step()

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    return proba, preds


# ===== Evaluation =====

feature_sets = {
    '10 raw bands': X_raw,
    '6 SR indices': sr_features,
    '6 SR + 7 classical': sr_classical,
}

models = {
    'MLP': MLP,
    '1D-CNN': Conv1DNet,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = {}

for model_name, model_class in models.items():
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    for feat_name, X_feat in feature_sets.items():
        n_input = X_feat.shape[1]
        fold_aucs = {c: [] for c in range(n_classes)}
        fold_ba = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_feat, y_mapped)):
            X_tr, X_te = X_feat[train_idx], X_feat[test_idx]
            y_tr, y_te = y_mapped[train_idx], y_mapped[test_idx]

            proba, preds = train_and_eval(
                model_class, X_tr, y_tr, X_te, y_te,
                n_input, n_classes, epochs=30, batch_size=512
            )

            fold_ba.append(balanced_accuracy_score(y_te, preds))

            for ci in range(n_classes):
                y_bin = (y_te == ci).astype(int)
                if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                    auc = roc_auc_score(y_bin, proba[:, ci])
                    fold_aucs[ci].append(auc)

        mean_auc = np.mean([np.mean(fold_aucs[ci]) for ci in range(n_classes) if fold_aucs[ci]])
        mean_ba = np.mean(fold_ba)

        key = f"{model_name} | {feat_name}"
        result = {
            'model': model_name,
            'features': feat_name,
            'n_features': n_input,
            'n_params': sum(p.numel() for p in model_class(n_input, n_classes).parameters()),
            'ba': float(mean_ba),
            'ba_std': float(np.std(fold_ba)),
            'mean_auc': float(mean_auc),
            'per_class_auc': {
                str(class_ids[ci]): float(np.mean(fold_aucs[ci]))
                for ci in range(n_classes) if fold_aucs[ci]
            },
        }
        all_results[key] = result

        CLASS_NAMES = {0: "Silic", 1: "AdvArg", 2: "ArgPh", 3: "Prop", 4: "IrOx", 5: "PotSk"}
        per_class_str = "  ".join(
            f"{CLASS_NAMES.get(ci, '?')}:{np.mean(fold_aucs[ci]):.3f}"
            for ci in range(n_classes) if fold_aucs[ci]
        )
        print(f"  {feat_name:25s} BA={mean_ba:.3f}±{np.std(fold_ba):.3f}  mAUC={mean_auc:.3f}  params={result['n_params']}  {per_class_str}")


# ===== Summary =====
print(f"\n{'='*70}")
print("SUMMARY — DL vs RF (from previous results)")
print(f"{'='*70}")

# Load RF results for comparison
rf_data = json.load(open(RESULTS_DIR / "sr_rf_ensemble.json"))

print(f"{'Method':<30} {'Features':<25} {'Params':>7} {'BA':>6} {'mAUC':>6}")
print("-" * 80)

# RF rows
for k, v in rf_data.items():
    if 'RF' not in k:
        continue
    feat = k.split(': ')[1] if ': ' in k else k
    print(f"{'RF (200 trees)':<30} {feat:<25} {'~50K':>7} {v['balanced_accuracy']['mean']:>6.3f} {v['mean_auc']:>6.3f}")

print("-" * 80)

# DL rows
for k, v in all_results.items():
    print(f"{v['model']:<30} {v['features']:<25} {v['n_params']:>7} {v['ba']:>6.3f} {v['mean_auc']:>6.3f}")

# Save
output = RESULTS_DIR / "dl_comparison.json"
with open(output, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {output}")

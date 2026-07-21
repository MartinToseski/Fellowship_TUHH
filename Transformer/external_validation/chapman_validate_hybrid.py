from pathlib import Path
import sys

import numpy as np
import torch

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, multilabel_confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.CNN_Transformer_Run import ECGLitModule, Config
from chapman_preprocessing import load_external_validation


SUPERCLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]


class ExternalDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    probs = []
    labels = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())

    return (np.concatenate(probs), np.concatenate(labels))


def evaluate(probs, labels, thresholds, class_names):
    pred = probs >= thresholds

    auc_macro = roc_auc_score(
        labels,
        probs,
        average="macro"
    )

    f1_macro = f1_score(
        labels,
        pred,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(
        labels.flatten(),
        pred.flatten()
    )

    # ===========================
    # Per-class metrics
    # ===========================
    precision = precision_score(
        labels,
        pred,
        average=None,
        zero_division=0
    )

    recall = recall_score(
        labels,
        pred,
        average=None,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        pred,
        average=None,
        zero_division=0
    )

    auc = roc_auc_score(
        labels,
        probs,
        average=None
    )

    cms = multilabel_confusion_matrix(labels, pred)

    print("\n")
    print("=" * 60)
    print("EXTERNAL VALIDATION RESULTS")
    print("=" * 60)

    print("\nOverall")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {f1_macro:.4f}")
    print(f"Macro AUC: {auc_macro:.4f}")

    print("\nPer-class metrics")
    print(
        f"{'Superclass':<12}"
        f"{'Precision':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'AUC':>12}"
    )

    for i, name in enumerate(class_names):
        print(
            f"{name:<6}|"
            f"{precision[i]:>12.3f}"
            f"{recall[i]:>12.3f}"
            f"{f1[i]:>12.3f}"
            f"{auc[i]:>12.3f}"
        )

    print("\nPer-class confusion matrix:")

    for name, cm in zip(class_names, cms):
        tn, fp, fn, tp = cm.ravel()

        pos_total = tp + fn
        neg_total = tn + fp

        tp_pct = 100 * tp / pos_total if pos_total else 0
        fn_pct = 100 * fn / pos_total if pos_total else 0

        fp_pct = 100 * fp / neg_total if neg_total else 0
        tn_pct = 100 * tn / neg_total if neg_total else 0

        print()
        print(f"CONFUSION MATRIX - {name}")
        print("                                    Predicted")
        print("                            Positive          Negative ")
        print(
            f"Actual   Positive"
            f"{tp:10d} ({tp_pct:5.1f} %) "
            f"{fn:10d} ({fn_pct:5.1f} %)"
        )
        print(
            f"         Negative"
            f"{fp:10d} ({fp_pct:5.1f} %) "
            f"{tn:10d} ({tn_pct:5.1f} %)"
        )

    return {
        "accuracy": accuracy,
        "macro_auc": auc_macro,
        "macro_f1": f1_macro,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrices": cms,
    }


device = "cuda"
checkpoint = Path("logs/CNN+Transformer/d384_head8_lay6_ff2304_ptch4_plmean_poslearnable_dr0.1_lr0.0005_ep100_optadamw_pat15_patt0.0001_wd0.01_lossweighted_bce_actgelu_normpre_20260714_140251/checkpoints/epoch=26-val_f1_macro=0.7395-val_auc_macro=0.9214.ckpt")

config = Config(
    model_name="CNN+Transformer",

    sampling_rate=100,
    augmentation="both",

    batch_size=64,
    learning_rate=5e-4,
    weight_decay=0.01,
    dropout=0.1,

    d_model = 384,
    n_heads = 8,
    n_layers = 6,
    ff_dim = 2304,

    patch_size = 4,
    pooling="mean",

    positional_encoding="learnable",
    activation="gelu",
    loss="weighted_bce",
    norm_first=True,

    num_classes=5,
    max_epochs=100,
    warmup_epochs=10,
    threshold=0.5,

    patience=15,
    early_stop_threshold=1e-4,
    gradient_clip_val=1.0
)


model = ECGLitModule.load_from_checkpoint(checkpoint, config=config, strict=False)
model.to(device)

X, y = load_external_validation()

print("NaNs in X:", np.isnan(X).sum())
print("Infs in X:", np.isinf(X).sum())

print("Min:", np.nanmin(X))
print("Max:", np.nanmax(X))

print(f"External ECGs: {len(X)}")

mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
y = mlb.fit_transform(y)

loader = torch.utils.data.DataLoader(ExternalDataset(X, y), batch_size=64, shuffle=False)

probs, labels = predict(model, loader, device)
results = evaluate(probs, labels, thresholds=0.5, class_names=SUPERCLASSES)

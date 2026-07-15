from pathlib import Path
import sys

import numpy as np
import torch

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, multilabel_confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.CNN1d.CNN1d_Run import ECGLitModule, Config
from chapman_subset_preprocessing import load_external_validation


SUPERCLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]


class ExternalDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        X = np.transpose(X, (0, 2, 1))
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
checkpoint = Path("logs/ModernCNN/ks7_dr0.3_lr0.001_rate100_epochs50_adam_20260706_102829/checkpoints/epoch=11-val_auc_macro=0.9302.ckpt")

config = Config(
    model_name="1dCNN_Run5",

    sampling_rate=100,
    augmentation="both",

    batch_size=256,
    learning_rate=1e-3,
    weight_decay=0.0,

    kernel_size=7,
    dropout=0.3,

    optimizer="adam",

    num_classes=5,
    max_epochs=50,
    threshold=0.5,
)


print(__file__)
print(checkpoint)

model = ECGLitModule.load_from_checkpoint(checkpoint, config=config, pos_weight=torch.ones(config.num_classes))
model.to(device)

X, y = load_external_validation()
print(f"External ECGs: {len(X)}")

mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
y = mlb.fit_transform(y)

loader = torch.utils.data.DataLoader(ExternalDataset(X, y), batch_size=64, shuffle=False)

probs, labels = predict(model, loader, device)
results = evaluate(probs, labels, thresholds=[0.32, 0.45, 0.34, 0.45, 0.35], class_names=SUPERCLASSES)

np.save("cache/chapman_probs.npy", probs)
np.save("cache/chapman_labels.npy", labels)
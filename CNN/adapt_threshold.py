from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import f1_score

from CNN1d_Modern import ECGLitModule, ECGDataModule, Config

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, multilabel_confusion_matrix, recall_score


CLASS_NAMES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]


def find_best_thresholds(probs, labels):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best = []

    for c in range(labels.shape[1]):
        best_thr = 0.5
        best_f1 = -1

        for thr in thresholds:
            pred = probs[:, c] >= thr

            f1 = f1_score(
                labels[:, c],
                pred,
                zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        best.append(best_thr)

    return np.array(best)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    labels = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())

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
    print("ADAPTIVE THRESHOLD RESULTS")
    print("=" * 60)

    print("\nOverall")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {f1_macro:.4f}")
    print(f"Macro AUC: {auc_macro:.4f}")

    print("\nThresholds")
    for name, thr in zip(class_names, thresholds):
        print(f"{name:6s}: {thr:.2f}")

    print("\nPer-class metrics")
    print(
        f"{'Class':<8}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'AUC':>12}"
    )

    for i, name in enumerate(class_names):
        print(
            f"{name:<8}"
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
checkpoint = Path("logs/ModernCNN/ks3_dr0.5_lr0.001_rate100_epochs50_adam_20260706_100403/checkpoints/epoch=10-val_auc_macro=0.9327.ckpt")

config = Config(
    sampling_rate=100,
    batch_size=64,      # doesn't affect inference much
    learning_rate=1e-3, # not used
    kernel_size=3,
    dropout=0.5,
    weight_decay=0.0,
    optimizer="adam",
    model_name="ModernCNN_test",
    max_epochs=50
)
model = ECGLitModule.load_from_checkpoint(checkpoint, config=config)
model.to(device)

data = ECGDataModule(config)
data.setup()

val_probs, val_labels = predict(model, data.val_dataloader(), device)
best_thresholds = find_best_thresholds(val_probs, val_labels)
print(best_thresholds)

test_probs, test_labels = predict(model, data.test_dataloader(), device)
results = evaluate(test_probs, test_labels, best_thresholds, CLASS_NAMES)

print()
print("Adaptive thresholds")
print(best_thresholds)
print()
print(f"Macro AUC : {results['macro_auc']:.4f}")
print(f"Macro F1  : {results['macro_f1']:.4f}")
print(f"Accuracy  : {results['accuracy']:.4f}")

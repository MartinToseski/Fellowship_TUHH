from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import f1_score

from models.CNN1d.CNN1d_GridSearch import ECGLitModule, ECGDataModule, Config

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, multilabel_confusion_matrix, recall_score


CLASS_NAMES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]


def find_best_thresholds(probs, labels, coarse_step=0.05, fine_step=0.005, max_iterations=10, tolerance=1e-4):
    n_classes = labels.shape[1]
    thresholds = np.full(n_classes, 0.5)
    previous_macro_f1 = -1.0

    for iteration in range(max_iterations):
        print(f"\nCoordinate Descent Iteration {iteration + 1}")

        # Optimize every class separately while fixing the others
        for c in range(n_classes):
            # Stage 1 : Coarse Search
            coarse_thresholds = np.arange(0.05, 0.951, coarse_step)

            best_thr = thresholds[c]
            best_score = -1

            for thr in coarse_thresholds:
                candidate = thresholds.copy()
                candidate[c] = thr

                pred = probs >= candidate

                macro_f1 = f1_score(
                    labels,
                    pred,
                    average="macro",
                    zero_division=0,
                )

                if macro_f1 > best_score:
                    best_score = macro_f1
                    best_thr = thr

            # Stage 2 : Fine Search
            low = max(0.05, best_thr - coarse_step)
            high = min(0.95, best_thr + coarse_step)

            fine_thresholds = np.arange(
                low,
                high + fine_step,
                fine_step,
            )

            for thr in fine_thresholds:
                candidate = thresholds.copy()
                candidate[c] = thr

                pred = probs >= candidate

                macro_f1 = f1_score(
                    labels,
                    pred,
                    average="macro",
                    zero_division=0,
                )

                if macro_f1 > best_score:
                    best_score = macro_f1
                    best_thr = thr

            thresholds[c] = round(float(best_thr), 3)

        # Evaluate current solution
        pred = probs >= thresholds

        macro_f1 = f1_score(
            labels,
            pred,
            average="macro",
            zero_division=0,
        )

        print(f"Macro F1 : {macro_f1:.4f}")
        print(f"Thresholds : {thresholds}")

        # Convergence
        if abs(macro_f1 - previous_macro_f1) < tolerance:
            print("\nCoordinate Descent converged.")
            break

        previous_macro_f1 = macro_f1

    print("\nFinal optimized thresholds:")
    for cls, thr in zip(CLASS_NAMES, thresholds):
        print(f"{cls:6s}: {thr:.3f}")

    print(f"\nValidation Macro F1: {previous_macro_f1:.4f}")
    return thresholds


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
checkpoint = Path("../logs/ModernCNN/ks7_dr0.3_lr0.001_rate100_epochs50_adam_20260706_102829/checkpoints/epoch=11-val_auc_macro=0.9302.ckpt")

config = Config(
    sampling_rate=100,
    batch_size=256,      # doesn't affect inference much
    learning_rate=1e-3, # not used
    kernel_size=7,
    dropout=0.3,
    weight_decay=0.0,
    optimizer="adam",
    model_name="ModernCNN_Threshold",
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

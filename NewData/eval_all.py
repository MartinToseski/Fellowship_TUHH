import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, multilabel_confusion_matrix

import heuristic_fill
import eval_V1V6
import eval_Generator

from CNN1d_GridSearch import ECGLitModule, ECGDataModule, Config, SUPERCLASSES


# =============================================================================
# CHECKPOINTS
# =============================================================================

BASELINE_CHECKPOINT = ("logs/C_ks7_dr0.3_lr0.001_rate100_epochs50_adam_20260706_102829/checkpoints/epoch=11-val_auc_macro=0.9302.ckpt")
CNN6_CHECKPOINT = ("logs/V1-V6_CNN/ks7_dr0.3_lr0.001_rate100_epochs50_adam_20260716_130204/checkpoints/epoch=8-val_auc_macro=0.9090.ckpt")
GENERATOR_CHECKPOINT = ("logs/Lead_Reconstruction_CNN/dr0.3_lr0.001_rate100_epochs100_adam_20260716_142548/checkpoints/epoch=42-val_loss=0.1340.ckpt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# BASELINE MODEL
# =============================================================================
config = Config(
    sampling_rate=100,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=0.0,
    kernel_size=7,
    dropout=0.3,
    augmentation="both",
    optimizer="adam",
    num_classes=5,
    max_epochs=50,
    threshold=0.5,
)

baseline = ECGLitModule.load_from_checkpoint(BASELINE_CHECKPOINT, config=config)
baseline.to(DEVICE)
baseline.eval()


# =============================================================================
# METHOD NAMES
# =============================================================================
METHODS = [
    "Original",
    "V1V6",
    "Zero",
    "Mean",
    "Duplicate",
    "KNN",
    "Linear Regression",
    "Generator",
]

heuristic_methods = {
    "zero": "Zero",
    "mean": "Mean",
    "duplicate": "Duplicate",
    "knn": "KNN",
    "linear_regression": "Linear Regression",
}


# =============================================================================
# RESULT STORAGE
# =============================================================================
results = {
    method: {
        "labels": [],
        "probabilities": [],
        "predictions": [],
    }
    for method in METHODS
}


# =============================================================================
# BASELINE PREDICTION
# =============================================================================
@torch.no_grad()
def predict_original(signal):
    signal = signal.to(DEVICE)

    logits = baseline(signal)
    probs = torch.sigmoid(logits)
    preds = (probs >= config.threshold).int()

    return (
        logits.cpu().numpy(),
        probs.cpu().numpy(),
        preds.cpu().numpy(),
    )


def update_results(method, labels, probs, preds):
    results[method]["labels"].append(labels.astype(np.int32))
    results[method]["probabilities"].append(probs.astype(np.float32))
    results[method]["predictions"].append(preds.astype(np.int32))


def compute_metrics(labels, probs, preds):
    metrics = {
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(
            labels,
            preds,
            average="macro",
            zero_division=0,
        ),
        "Recall": recall_score(
            labels,
            preds,
            average="macro",
            zero_division=0,
        ),
        "Macro F1": f1_score(
            labels,
            preds,
            average="macro",
            zero_division=0,
        ),
    }

    try:
        metrics["Macro AUROC"] = roc_auc_score(
            labels,
            probs,
            average="macro",
        )
    except ValueError:
        metrics["Macro AUROC"] = np.nan

    return metrics


def print_per_class_metrics(method, labels, probs, preds):
    rows = []

    for i, cls in enumerate(SUPERCLASSES):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = np.nan

        rows.append({
            "Class": cls,
            "Precision": precision_score(
                labels[:, i],
                preds[:, i],
                zero_division=0,
            ),
            "Recall": recall_score(
                labels[:, i],
                preds[:, i],
                zero_division=0,
            ),
            "F1": f1_score(
                labels[:, i],
                preds[:, i],
                zero_division=0,
            ),
            "AUROC": auc,
        })

    df = pd.DataFrame(rows)

    print("\n")
    print("=" * 90)
    print(f"{method} PER-CLASS RESULTS")
    print("=" * 90)
    print(df.round(4).to_string(index=False))

    return df


def summarize_results():
    rows = []

    for method in METHODS:
        labels = np.concatenate(results[method]["labels"], axis=0)
        probs = np.concatenate(results[method]["probabilities"], axis=0)
        preds = np.concatenate(results[method]["predictions"], axis=0)

        metrics = compute_metrics(labels, probs, preds)
        metrics["Method"] = method
        rows.append(metrics)

        # Detailed evaluation only for V1V6 and Generator
        if method in ["V1V6", "Generator"]:
            print_per_class_metrics(
                method,
                labels,
                probs,
                preds,
            )

            cms = multilabel_confusion_matrix(labels, preds)

            print("Confusion Matrices:")
            for cls, cm in zip(SUPERCLASSES, cms):
                print(f"\n{cls}")
                print(pd.DataFrame(
                    cm,
                    index=["Actual Negative", "Actual Positive"],
                    columns=["Predicted Negative", "Predicted Positive"],
                ))

    df = pd.DataFrame(rows)
    df = df[
        [
            "Method",
            "Accuracy",
            "Precision",
            "Recall",
            "Macro F1",
            "Macro AUROC",
        ]
    ]

    print("\nOverall Results")
    print(df.round(4))

    return df


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    data = ECGDataModule(config)
    data.setup()

    test_loader = data.test_dataloader()

    print(f"Evaluating {len(test_loader.dataset)} PTB-XL test ECGs...\n")

    for signals, labels in tqdm(test_loader):
        labels_np = labels.numpy()

        _, probs, preds = predict_original(signals)
        update_results("Original", labels_np, probs, preds)

        for signal, label in zip(signals, labels_np):
            signal = signal.cpu().numpy().T      # (1000,12)
            chest = signal[:, 6:]                # (1000,6)
            label = label[None]

            _, probs, preds = eval_V1V6.predict_signal(
                chest,
                preprocess=False,
            )
            update_results("V1V6", label, probs, preds)

            for method, name in heuristic_methods.items():
                _, probs, preds = heuristic_fill.predict_method(
                    chest,
                    method,
                    preprocess=False,
                )
                update_results(name, label, probs, preds)

            _, probs, preds = eval_Generator.predict_signal(
                chest,
                preprocess=False,
            )
            update_results("Generator", label, probs, preds)

    df = summarize_results()

    print("\nEvaluation complete.")
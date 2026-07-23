from pathlib import Path
import sys

import numpy as np
import torch
import copy
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, multilabel_confusion_matrix
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from validation.cnn_train_chapman import ECGLitModule, Config
from preprocessing.preprocessing import split_data, per_lead_global_normalization
from utils.utils import remove_empty_diagnosis


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
    print("FINE-TUNE ON PTBXL RESULTS")
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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validation_f1(model, loader, device):
    model.eval()

    probs = []
    labels = []

    for x, y in loader:
        x = x.to(device)

        logits = model(x)

        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    pred = probs >= 0.5

    macro_f1 = f1_score(
        labels,
        pred,
        average="macro",
        zero_division=0,
    )

    return macro_f1


device = "cuda"
checkpoint = Path("logs/CNN_Chapman_Train/")

config = Config(
    model_name="CNN_PTBXL_Fine_Tune",

    sampling_rate=100,
    augmentation="both",

    batch_size=256,
    learning_rate=1e-3,
    weight_decay=0.0,

    kernel_size=7,
    dropout=0.3,

    optimizer="adam",

    num_classes=5,
    max_epochs=100,
    threshold=0.5,
)


print(__file__)
print(checkpoint)

model = ECGLitModule.load_from_checkpoint(checkpoint, config=config, pos_weight=torch.ones(config.num_classes))
model.to(device)

for p in model.parameters():
    p.requires_grad = True

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam([
    {"params": model.features.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4},
])

X_train, y_train, X_val, y_val, X_test, y_test = split_data(100)

X_train, y_train = remove_empty_diagnosis(X_train, y_train)
X_val, y_val = remove_empty_diagnosis(X_val, y_val)
X_test, y_test = remove_empty_diagnosis(X_test, y_test)

# label encoding
mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
y_train = mlb.fit_transform(y_train)
y_val = mlb.transform(y_val)
y_test = mlb.transform(y_test)

# normalization
X_train, X_val, X_test = per_lead_global_normalization(X_train, X_val, X_test)

train_loader = torch.utils.data.DataLoader(
    ExternalDataset(X_train, y_train),
    batch_size=config.batch_size,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    ExternalDataset(X_val, y_val),
    batch_size=config.batch_size,
    shuffle=False,
)

test_loader = torch.utils.data.DataLoader(
    ExternalDataset(X_test, y_test),
    batch_size=config.batch_size,
    shuffle=False,
)

best_f1 = -1.0
best_state = None
history = []

patience = 15
patience_counter = 0
improvement_threshold = 1e-4

for epoch in range(config.max_epochs):
    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
    )

    val_f1 = validation_f1(
        model,
        val_loader,
        device,
    )
    
    improved = val_f1 > best_f1 + improvement_threshold

    if improved:
        best_f1 = val_f1
        best_state = copy.deepcopy(model.state_dict())
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_f1,
                "config": config,
            },
            "logs/ptbxl_best_model.pth",
        )
        patience_counter = 0
    else:
        patience_counter += 1

    status = "✓" if improved else ""
    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Macro F1: {val_f1:.4f} | "
        f"Best: {best_f1:.4f} {status}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_f1_macro": val_f1,
        "best_val_f1": best_f1,
        "is_best": abs(val_f1 - best_f1) < 1e-12,
    })

    if patience_counter >= patience:
        print(f"\nEarly stopping after {epoch+1} epochs.")
        break


model.load_state_dict(best_state)
print(f"\nBest validation Macro F1: {best_f1:.4f}")

probs, labels = predict(model, test_loader, device)
results = evaluate(probs, labels, thresholds=0.5, class_names=SUPERCLASSES)

history = pd.DataFrame(history)
history.to_csv(
    "logs/ptbxl_finetune_metrics.csv",
    index=False,
)

report = {
    "best_val_f1": best_f1,
    "test_accuracy": results["accuracy"],
    "test_macro_f1": results["macro_f1"],
    "test_macro_auc": results["macro_auc"],
}
pd.DataFrame([report]).to_csv(
    "logs/ptbxl_test_results.csv",
    index=False,
)

np.save("cache/ptbxl_full_probs.npy", probs)
np.save("cache/ptbxl_full_labels.npy", labels)
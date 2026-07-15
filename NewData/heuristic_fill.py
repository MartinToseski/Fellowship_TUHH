import numpy as np
import torch
import joblib

from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from CNN1d_GridSearch import ECGLitModule, ECGDataModule, Config

SUPERCLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]

_regression = None
_knn = None
_knn_limb = None

WINDOW = 1000
STRIDE = 250
THRESHOLDS = np.array([0.32, 0.45, 0.34, 0.45, 0.35])

FILL_METHOD = "duplicate"
AGGREGATION = "median"


device = "cuda"
checkpoint = Path("logs/C_ks7_dr0.3_lr0.001_rate100_epochs50_adam_20260706_102829/checkpoints/epoch=11-val_auc_macro=0.9302.ckpt")

config = Config(
    sampling_rate=100,
    batch_size=256,      # doesn't affect inference much
    learning_rate=1e-3, # not used
    kernel_size=7,
    dropout=0.3,
    weight_decay=0.0,
    optimizer="adam",
    model_name="ModernCNN",
    max_epochs=50
)

model = ECGLitModule.load_from_checkpoint(checkpoint, config=config)
model.to(device)

data = ECGDataModule(config)
data.setup()



def load_precordial_leads(folder):
    return (
        np.loadtxt(f"{folder}/V1.Session 0 - Page 1.7.TXT"),
        np.loadtxt(f"{folder}/V2.Session 0 - Page 1.8.TXT"),
        np.loadtxt(f"{folder}/V3.Session 0 - Page 1.9.TXT"),
        np.loadtxt(f"{folder}/V4.Session 0 - Page 1.10.TXT"),
        np.loadtxt(f"{folder}/V5.Session 0 - Page 1.11.TXT"),
        np.loadtxt(f"{folder}/V6.Session 0 - Page 1.12.TXT"),
    )


def normalize(signal):
    mean = np.load("cache/ptbxl_perlead_mean.npy")
    std = np.load("cache/ptbxl_perlead_std.npy")
    return (signal - mean) / std


def fill_missing_leads(v1, v2, v3, v4, v5, v6, method="zero"):
    if method == "zero":
        zero = np.zeros_like(v1)
        return np.stack([zero, zero, zero, zero, zero, zero, v1, v2, v3, v4, v5, v6], axis=1)
    
    elif method == "duplicate":
        return np.stack([v1, v2, v3, v4, v5, v6, v1, v2, v3, v4, v5, v6], axis=1)

    elif method == "mean":
        m = (v1 + v2 + v3 + v4 + v5 + v6) / 6
        return np.stack([m, m, m, m, m, m, v1, v2, v3, v4, v5, v6], axis=1)
    
    elif method == "linear_regression":
        limb = predict_linear_regression(v1, v2, v3, v4, v5, v6)
        return np.concatenate(
            [limb, np.stack([v1,v2,v3,v4,v5,v6], axis=1)],
            axis=1,
        )   
    
    elif method == "knn":
        limb = predict_knn(v1, v2, v3, v4, v5, v6)
        return np.concatenate(
            [limb, np.stack([v1,v2,v3,v4,v5,v6], axis=1)],
            axis=1,
        )   


def print_results(logits, probs, prediction):
    print()
    print("="*60)
    print("MODEL PREDICTION")
    print("="*60)

    for cls,p,t in zip(SUPERCLASSES, probs, prediction):
        print(f"{cls:6s}   probability={p:.4f}   predicted={bool(t)}")

    diagnosis = []
    for cls,pred in zip(SUPERCLASSES,prediction):
        if pred:
            diagnosis.append(cls)

    if len(diagnosis)==0:
        diagnosis.append("No class exceeded threshold")

    print()
    print("Predicted diagnosis")
    for d in diagnosis:
        print("-", d)

    print()
    for cls,logit,p in zip(SUPERCLASSES, logits.cpu().numpy()[0], probs):
        print(f"{cls:6s}   logit={logit:8.3f}   prob={p:.3f}")

    print()
    print("=" * 60)
    print("AGGREGATED PROBABILITIES")
    print("=" * 60)

    for cls, p in zip(SUPERCLASSES, probs):
        print(f"{cls:6s}   {p:8.4f}")

    print()
    print("=" * 60)
    print("WINDOW SUMMARY")
    print("=" * 60)

    for window in window_predictions:
        print()
        print(
            f"Samples "
            f"{window['start']:6d}"
            f" - "
            f"{window['end']-1:6d}"
        )

        for cls, p in zip(SUPERCLASSES, window["probs"]):
            print(f"{cls:6s}   {p:8.3f}")


def predict_linear_regression(v1,v2,v3,v4,v5,v6):
    global _regression

    if _regression is None:
        _regression = joblib.load("cache/lead_regression.joblib")

    chest=np.stack([v1, v2, v3, v4, v5, v6], axis=1)
    pred = _regression.predict(chest.reshape(1,-1))
    return pred.reshape(len(v1), 6)


def predict_knn(v1,v2,v3,v4,v5,v6):
    global _knn
    global _knn_limb

    if _knn is None:
        chest=np.load("cache/knn_chest.npy")
        _knn_limb=np.load("cache/knn_limb.npy")
        _knn=NearestNeighbors(n_neighbors=5, metric="euclidean")
        _knn.fit(chest)

    query=np.stack(
        [v1,v2,v3,v4,v5,v6],
        axis=1,
    ).reshape(1,-1)

    _, idx = _knn.kneighbors(query)
    return _knn_limb[idx[0]].mean(axis=0)


def aggregate(probabilities, method):
    probabilities = np.asarray(probabilities)
    if method == "mean":
        return probabilities.mean(axis=0)
    elif method == "max":
        return probabilities.max(axis=0)
    elif method == "median":
        return np.median(probabilities, axis=0)
    raise ValueError(method)


def create_windows(signal):
    windows = []
    start = 0

    while start + WINDOW <= len(signal):
        windows.append((start, signal[start:start + WINDOW].copy()))
        start += STRIDE

    last_start = len(signal) - WINDOW

    if windows[-1][0] != last_start:
        windows.append((last_start, signal[last_start:].copy()))

    return windows


def predict_window(signal):
    signal = normalize(signal)
    signal = np.transpose(signal, (1, 0))
    signal = (torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device))

    model.eval()
    with torch.no_grad():
        logits = model(signal)
        probs = torch.sigmoid(logits)

    return (logits.cpu().numpy()[0], probs.cpu().numpy()[0])


v1, v2, v3, v4, v5, v6 = load_precordial_leads("txt_files")
signal = fill_missing_leads(v1, v2, v3, v4, v5, v6, method=FILL_METHOD)


window_probs = []
window_logits = []
window_predictions = []

windows = create_windows(signal)

print()
print("=" * 60)
print("RECORDING INFORMATION")
print("=" * 60)
print(f"Samples        : {len(signal)}")
print(f"Window size    : {WINDOW}")
print(f"Stride         : {STRIDE}")
print(f"Num windows    : {len(windows)}")
print(f"Lead filling   : {FILL_METHOD}")
print(f"Aggregation    : {AGGREGATION}")

for i, (start, window) in enumerate(windows):
    logits, probs = predict_window(window)
    window_logits.append(logits)
    window_probs.append(probs)

    window_predictions.append({
        "start": start,
        "end": start + WINDOW,
        "probs": probs,
        "logits": logits,
    })

    print()
    print(f"Window {i+1:03d}")
    print(f"Samples {start:6d} - {start + WINDOW - 1}")

    for cls, p in zip(SUPERCLASSES, probs):
        print(f"{cls:6s}   {p:8.3f}")


probs = aggregate(window_probs, AGGREGATION)
logits = aggregate(window_logits, AGGREGATION)
prediction = probs >= THRESHOLDS

print_results(torch.tensor(logits).unsqueeze(0), probs, prediction)

import numpy as np
import torch
import joblib

from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from CNN1d_GridSearch import ECGLitModule, Config
from scipy.signal import resample_poly


_regression = None
_knn = None
_knn_limb = None

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CHANNEL_RESOLUTION_MV = 78e-6
FILL_METHODS = ["zero", "mean", "duplicate", "knn", "linear_regression"]
THRESHOLDS = np.array([0.32, 0.45, 0.34, 0.45, 0.35])
INPUT_LENGTH = 1000
WINDOW_RADIUS = 10

device = "cuda"
checkpoint = Path("logs/C_ks7_dr0.3_lr0.001_rate100_epochs50_adam_20260706_102829/checkpoints/epoch=11-val_auc_macro=0.9302.ckpt")

config = Config(
    sampling_rate=100,
    batch_size=256,      
    learning_rate=1e-3, 
    kernel_size=7,
    dropout=0.3,
    weight_decay=0.0,
    optimizer="adam",
    model_name="ModernCNN",
    max_epochs=50
)

model = ECGLitModule.load_from_checkpoint(checkpoint, config=config)
model.to(device)


def load_precordial_leads(folder):
    return (
        np.loadtxt(f"{folder}/V1.Session 0 - Page 1.7.TXT"),
        np.loadtxt(f"{folder}/V2.Session 0 - Page 1.8.TXT"),
        np.loadtxt(f"{folder}/V3.Session 0 - Page 1.9.TXT"),
        np.loadtxt(f"{folder}/V4.Session 0 - Page 1.10.TXT"),
        np.loadtxt(f"{folder}/V5.Session 0 - Page 1.11.TXT"),
        np.loadtxt(f"{folder}/V6.Session 0 - Page 1.12.TXT"),
    )


def resample(signal):
    return resample_poly(signal, up=1, down=20, axis=0)


def normalize(signal):
    mean = np.load("cache/ptbxl_perlead_mean.npy").reshape(-1)
    std = np.load("cache/ptbxl_perlead_std.npy").reshape(-1)
    return (signal - mean) / std


def fill_missing_leads(chest, method="zero"):
    if method == "zero":
        limb = np.zeros_like(chest)

    elif method == "duplicate":
        limb = chest.copy()

    elif method == "mean":
        m = chest.mean(axis=1, keepdims=True)
        limb = np.repeat(m, 6, axis=1)

    elif method == "linear_regression":
        limb = predict_linear_regression(chest)

    elif method == "knn":
        limb = predict_knn(chest)

    else:
        raise ValueError(method)

    return np.concatenate([limb, chest], axis=1)


def predict_linear_regression(chest):
    global _regression

    if _regression is None:
        _regression = joblib.load("cache/lead_regression.joblib")

    chest = np.pad(
        chest,
        ((WINDOW_RADIUS, WINDOW_RADIUS), (0, 0)),
        mode="reflect",
    )

    prediction = np.empty((len(chest) - 2 * WINDOW_RADIUS, 6), dtype=np.float32)

    for t in range(prediction.shape[0]):
        patch = chest[t:t + 2 * WINDOW_RADIUS + 1]
        prediction[t] = _regression.predict(
            patch.reshape(1, -1)
        )[0]

    return prediction


def predict_knn(chest):
    global _knn
    global _knn_limb

    if _knn is None:
        database = np.load("cache/knn_chest.npy")
        _knn_limb = np.load("cache/knn_limb.npy")
        _knn = NearestNeighbors(n_neighbors=5, metric="euclidean",)
        _knn.fit(database)

    query = chest.reshape(1, -1)
    _, idx = _knn.kneighbors(query)
    return _knn_limb[idx[0]].mean(axis=0)


def center_crop(signal):
    if len(signal) < INPUT_LENGTH:
        raise ValueError(
            f"Signal too short ({len(signal)} samples). "
            f"Expected at least {INPUT_LENGTH}."
        )

    start = (len(signal) - INPUT_LENGTH) // 2
    end = start + INPUT_LENGTH
    return signal[start:end].copy()


def predict(signal):
    signal = normalize(signal)
    signal = np.transpose(signal, (1, 0))

    signal = torch.tensor(
        signal,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(signal)
        probs = torch.sigmoid(logits)

    return (
        logits.cpu().numpy()[0],
        probs.cpu().numpy()[0],
    )


def print_results(logits, probs, prediction, fill_method):
    print()
    print(f"MODEL PREDICTION -> ({fill_method})")

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


for method in FILL_METHODS:
    
    v1, v2, v3, v4, v5, v6 = load_precordial_leads("txt_files")

    chest = np.stack([v1, v2, v3, v4, v5, v6], axis=1)
    chest *= CHANNEL_RESOLUTION_MV
    chest = resample(chest)
    chest = center_crop(chest)

    signal = fill_missing_leads(chest, method)

    logits, probs = predict(signal)
    prediction = probs >= THRESHOLDS

    print_results(
        torch.tensor(logits).unsqueeze(0),
        probs,
        prediction,
        method,
    )
    print("-" * 60)

import numpy as np
import torch

from pathlib import Path
from scipy.signal import resample_poly

from CNN1d_V1V6 import ECGLitModule, Config


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CHANNEL_RESOLUTION_MV = 78e-6
INPUT_LENGTH = 1000
THRESHOLDS = np.array([0.23,  0.54,  0.305, 0.51,  0.37])

device = "cuda"
checkpoint = Path("logs/V1-V6_CNN/ks7_dr0.3_lr0.001_rate100_epochs50_adam_20260716_130204/checkpoints/epoch=8-val_auc_macro=0.9090.ckpt")

config = Config(
    sampling_rate=100,
    batch_size=256,
    learning_rate=1e-3,
    kernel_size=7,
    dropout=0.3,
    weight_decay=0.0,
    optimizer="adam",
    model_name="ModernCNN",
    max_epochs=50,
)

model = ECGLitModule.load_from_checkpoint(checkpoint, config=config, map_location=device)
model.to(device)
model.eval()


def load_precordial_leads(folder):
    return np.stack(
        [
            np.loadtxt(f"{folder}/V1.Session 0 - Page 1.7.TXT"),
            np.loadtxt(f"{folder}/V2.Session 0 - Page 1.8.TXT"),
            np.loadtxt(f"{folder}/V3.Session 0 - Page 1.9.TXT"),
            np.loadtxt(f"{folder}/V4.Session 0 - Page 1.10.TXT"),
            np.loadtxt(f"{folder}/V5.Session 0 - Page 1.11.TXT"),
            np.loadtxt(f"{folder}/V6.Session 0 - Page 1.12.TXT"),
        ],
        axis=1,
    )


def resample(signal):
    return resample_poly(signal, up=1, down=20, axis=0)


def center_crop(signal):
    start = (len(signal) - INPUT_LENGTH) // 2
    return signal[start:start + INPUT_LENGTH]


def normalize(signal):
    mean = np.load("cache/ptbxl_perlead_mean.npy").reshape(-1)[6:]
    std = np.load("cache/ptbxl_perlead_std.npy").reshape(-1)[6:]
    return (signal - mean) / std


def predict(signal, normalize_input=True):
    if normalize_input:
        signal = normalize(signal)

    signal = torch.tensor(
        signal.T,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(signal)
        probs = torch.sigmoid(logits)

    return (
        logits.cpu().numpy()[0],
        probs.cpu().numpy()[0],
    )


def predict_signal(signal, preprocess=True):
    if preprocess:
        signal *= CHANNEL_RESOLUTION_MV
        signal = resample(signal)
        signal = center_crop(signal)
        logits, probs = predict(signal, normalize_input=True)
    else:
        logits, probs = predict(signal, normalize_input=False)

    prediction = probs >= THRESHOLDS
    return logits[None], probs[None], prediction[None]


def print_results(logits, probs, prediction):
    print()
    print("=" * 60)
    print("MODEL PREDICTION")
    print("=" * 60)

    diagnosis = []

    for cls, p, pred in zip(SUPERCLASSES, probs, prediction):
        print(f"{cls:6s} probability={p:.4f} predicted={bool(pred)}")
        if pred:
            diagnosis.append(cls)

    if not diagnosis:
        diagnosis = ["None"]

    print()
    print("Predicted diagnosis:")
    for d in diagnosis:
        print("-", d)

    print()

    for cls, logit, prob in zip(SUPERCLASSES, logits, probs):
        print(f"{cls:6s} logit={logit:8.3f} prob={prob:.3f}")


def run():
    signal = load_precordial_leads("txt_files")
    logits, probs = predict_signal(signal)
    prediction = probs >= THRESHOLDS
    print_results(logits, probs, prediction)


if __name__ == "__main__":
    run()
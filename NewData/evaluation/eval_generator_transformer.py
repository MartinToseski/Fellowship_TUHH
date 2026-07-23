import numpy as np
import torch
import sys

from pathlib import Path
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.CNN1d_LeadGenerator import ECGLitModule as Generator
from models.CNN1d_LeadGenerator import Config as GeneratorConfig

from models.Transformer_Run import ECGLitModule as Transformer
from models.Transformer_Run import Config as TransformerConfig


SUPERCLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]

CHANNEL_RESOLUTION_MV = 78e-6
INPUT_LENGTH = 1000

THRESHOLDS = np.array([0.68,  0.53,  0.395, 0.605, 0.605])

device = "cuda"


# ----------------------------------------------------------
# Lead Generator
# ----------------------------------------------------------

generator_ckpt = Path(
    "logs/Lead_Reconstruction_CNN/dr0.3_lr0.001_rate100_epochs100_adam_20260716_142548/checkpoints/epoch=42-val_loss=0.1340.ckpt"
)

generator_cfg = GeneratorConfig(sampling_rate=100)

generator = Generator.load_from_checkpoint(
    generator_ckpt,
    config=generator_cfg,
    map_location=device,
)

generator.to(device)
generator.eval()


# ----------------------------------------------------------
# Classifier
# ----------------------------------------------------------

transformer_ckpt = Path(
    "logs/T_d384_head8_lay6_ff2304_ptch4_plmean_poslearnable_dr0.1_lr0.0005_ep100_optadamw_pat15_patt0.0001_wd0.01_lossweighted_bce_actgelu_normpre_20260713_052757/checkpoints/epoch=57-val_f1_macro=0.7221-val_auc_macro=0.9125.ckpt"
)

transformer_cfg = TransformerConfig(
    model_name="Test",

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
    pooling = "mean",

    positional_encoding = "learnable",
    activation="gelu",
    loss="weighted_bce",
    norm_first=True,

    num_classes=5,
    max_epochs=100,
    threshold=0.5,
    warmup_epochs=10,

    patience=15,
    early_stop_threshold=1e-4,
    gradient_clip_val=1.0
)

transformer = Transformer.load_from_checkpoint(
    transformer_ckpt,
    config=transformer_cfg,
    map_location=device,
    strict=False
)

transformer.to(device)
transformer.eval()


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------

def load_precordial(folder):
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


def predict_signal(chest, preprocess=True):
    # ----------------------------------------------------------
    # Normalization
    # ----------------------------------------------------------
    mean12 = np.load("cache/ptbxl_perlead_mean.npy").reshape(-1)
    std12 = np.load("cache/ptbxl_perlead_std.npy").reshape(-1)

    mean6 = mean12[6:]
    std6 = std12[6:]

    # ----------------------------------------------------------
    # Generate limb leads
    # ----------------------------------------------------------
    if preprocess:
        chest_norm = (chest - mean6) / std6
    else:
        chest_norm = chest

    generator_input = torch.tensor(
        chest_norm.T,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_limb = generator(generator_input)

    predicted_limb = predicted_limb.cpu().numpy()[0].T

    # ----------------------------------------------------------
    # Build complete ECG
    # ----------------------------------------------------------
    ecg12 = np.concatenate(
        [
            predicted_limb,
            chest,
        ],
        axis=1,
    )

    # ----------------------------------------------------------
    # Normalize for classifier
    # ----------------------------------------------------------
    if preprocess:
        ecg12 = (ecg12 - mean12) / std12

    transformer_input = torch.tensor(
        ecg12,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------
    with torch.no_grad():
        logits = transformer(transformer_input)
        probs = torch.sigmoid(logits)

    logits = logits.cpu().numpy().squeeze()
    probs = probs.cpu().numpy().squeeze()
    prediction = probs >= THRESHOLDS

    return logits[None], probs[None], prediction[None]


def print_results(logits, probs, prediction):
    logits = np.squeeze(logits)
    probs = np.squeeze(probs)
    prediction = np.squeeze(prediction)
    print()
    print("=" * 60)
    print("LEAD GENERATOR + TRANSFORMER")
    print("=" * 60)

    diagnosis = []

    for cls, p, pred in zip(SUPERCLASSES, probs, prediction):
        print(f"{cls:6s} probability={p:.4f} predicted={bool(pred)}")
        if pred:
            diagnosis.append(cls)

    if len(diagnosis) == 0:
        diagnosis = ["None"]

    print()
    print("Diagnosis:")

    for d in diagnosis:
        print("-", d)

    print()

    for cls, logit, prob in zip(SUPERCLASSES, logits, probs):
        print(f"{cls:6s} logit={logit:8.3f} prob={prob:.3f}")


def run():
    chest = load_precordial("txt_files")

    chest *= CHANNEL_RESOLUTION_MV
    chest = resample(chest)
    chest = center_crop(chest)

    logits, probs, prediction = predict_signal(chest)

    print_results(logits, probs, prediction)


if __name__ == "__main__":
    run()
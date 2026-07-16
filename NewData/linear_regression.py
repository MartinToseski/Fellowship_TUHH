import joblib
import numpy as np

from sklearn.linear_model import Ridge
from preprocessing import load_dataset


WINDOW_RADIUS = 10          # 21-sample temporal context
ALPHA = 1.0
STEP = 2


print("Loading dataset...")
signals, _ = load_dataset(100)
print(f"Loaded {len(signals)} ECGs")


X = []
Y = []

print("Preparing training data...")

for ecg in signals:
    chest = ecg[:, 6:12]
    limb = ecg[:, :6]

    # reflect padding keeps edge morphology much better than zeros
    chest = np.pad(
        chest,
        ((WINDOW_RADIUS, WINDOW_RADIUS), (0, 0)),
        mode="reflect",
    )

    for t in range(0, limb.shape[0], STEP):
        patch = chest[t:t + 2 * WINDOW_RADIUS + 1]
        X.append(patch.reshape(-1))
        Y.append(limb[t])

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

print("Input :", X.shape)
print("Output:", Y.shape)

print("Training Ridge...")

model = Ridge(
    alpha=ALPHA,
    solver="lsqr",
)

model.fit(X, Y)

joblib.dump(model, "cache/lead_regression.joblib")

print("Saved.")
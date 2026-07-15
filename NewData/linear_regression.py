import joblib
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from pathlib import Path

from preprocessing import load_dataset   


WINDOW = 1000
STRIDE = 250


def create_windows(signal):
    windows = []
    for start in range(0, signal.shape[0] - WINDOW + 1, STRIDE):
        windows.append(signal[start:start + WINDOW])
    return windows


signals, _ = load_dataset()

X = []
Y = []

for ecg in signals:
    for window in create_windows(ecg):
        chest = window[:, 6:12]
        limb = window[:, :6]
        X.append(chest.reshape(-1))
        Y.append(limb.reshape(-1))

X = np.asarray(X)
Y = np.asarray(Y)

print(X.shape)
print(Y.shape)

model = MultiOutputRegressor(Ridge(alpha=1.0))
model.fit(X, Y)
joblib.dump(model, "cache/lead_regression.joblib")
print("Saved linear regression.")
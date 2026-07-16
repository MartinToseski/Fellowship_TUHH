import joblib
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

from preprocessing import load_dataset


signals, _ = load_dataset(100)

X = []
Y = []

for ecg in signals:
    chest = ecg[:, 6:12]
    limb = ecg[:, :6]

    X.append(chest.reshape(-1))
    Y.append(limb.reshape(-1))

X = np.asarray(X)
Y = np.asarray(Y)

print("Chest:", X.shape)
print("Limb :", Y.shape)

model = MultiOutputRegressor(Ridge(alpha=1.0), n_jobs=-1)
model.fit(X, Y)

joblib.dump(model, "cache/lead_regression.joblib")
print("Saved linear regression.")
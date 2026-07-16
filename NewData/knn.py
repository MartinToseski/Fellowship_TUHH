import numpy as np

from preprocessing import load_dataset


signals, _ = load_dataset(100)

database_chest = []
database_limb = []

for ecg in signals:
    database_chest.append(ecg[:, 6:12].reshape(-1))
    database_limb.append(ecg[:, :6])

database_chest = np.asarray(database_chest)
database_limb = np.asarray(database_limb)

print("Chest:", database_chest.shape)
print("Limb :", database_limb.shape)

np.save("cache/knn_chest.npy", database_chest)
np.save("cache/knn_limb.npy", database_limb)
print("Saved KNN database.")
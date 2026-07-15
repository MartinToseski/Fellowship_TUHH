import numpy as np

from pathlib import Path
from preprocessing import load_dataset   


WINDOW = 1000
STRIDE = 250


def create_windows(signal):
    windows = []
    for start in range(0, signal.shape[0]-WINDOW+1, STRIDE):
        windows.append(signal[start:start+WINDOW])
    return windows


signals,_ = load_dataset()

database_chest=[]
database_limb=[]

for ecg in signals:
    for window in create_windows(ecg):
        database_chest.append(window[:,6:12].reshape(-1))
        database_limb.append(window[:,:6])

database_chest=np.asarray(database_chest)
database_limb=np.asarray(database_limb)

np.save("cache/knn_chest.npy", database_chest)
np.save("cache/knn_limb.npy", database_limb)

print(database_chest.shape)
print(database_limb.shape)
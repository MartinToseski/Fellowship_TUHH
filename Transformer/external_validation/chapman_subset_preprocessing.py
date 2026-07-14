import pandas as pd
import numpy as np
import wfdb
import os
import pickle

from pathlib import Path
from scipy.signal import resample_poly
from collections import Counter


DATA_PATH = "../../data/chapman_shaoxing/"

CACHE_X_PATH = f"cache/chapman_X.npy"
CACHE_Y_PATH = f"cache/chapman_Y.pkl"

PTB_MEAN = "cache/ptbxl_perlead_mean.npy"
PTB_STD = "cache/ptbxl_perlead_std.npy"

ABBREV_TO_SUPERCLASS = {

    # ==========================================================
    # NORM
    # ==========================================================
    "NORM": "NORM",
    "NSR": "NORM",
    "SB": "NORM",
    "SA": "NORM",
    "SR": "NORM",
    "STach": "NORM",

    # ==========================================================
    # MI
    # ==========================================================
    "MI": "MI",
    "OldMI": "MI",

    # ==========================================================
    # ST-T CHANGES
    # ==========================================================
    "NSSTTA": "STTC",
    "STD": "STTC",
    "STE": "STTC",
    "TAb": "STTC",
    "TInv": "STTC",
    "LQT": "STTC",
    "UAb": "STTC",

    # ==========================================================
    # CONDUCTION / ARRHYTHMIA
    # ==========================================================
    "AF": "CD",
    "AFL": "CD",

    "AVB": "CD",
    "IAVB": "CD",
    "IIAVB": "CD",
    "IIIAVB": "CD",

    "RBBB": "CD",
    "LBBB": "CD",
    "IRBBB": "CD",

    "NSIVCB": "CD",

    "WPW": "CD",

    "SVT": "CD",
    "ATach": "CD",

    "VPB": "CD",
    "VBig": "CD",
    "VEsB": "CD",
    "VPEx": "CD",

    "JE": "CD",

    "CCR": "CD",
    "CR": "CD",

    "LAD": "CD",
    "RAD": "CD",

    "LQRSV": "CD",

    "PWC": "CD",

    # ==========================================================
    # HYPERTROPHY
    # ==========================================================
    "LVH": "HYP",
    "LVHV": "HYP",

    "RVH": "HYP",

    "RAH": "HYP",
    "RAHV": "HYP",

    "LAE": "HYP",
}

expected_leads = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6",
]

mapping = pd.concat(
    [
        pd.read_csv(f"{DATA_PATH}dx_mapping_scored.csv"),
        pd.read_csv(f"{DATA_PATH}dx_mapping_unscored.csv"),
    ],
    ignore_index=True,
)

mapping = mapping.drop_duplicates(subset="SNOMEDCTCode")

SNOMED_TO_ABBREV = dict(
    zip(
        mapping["SNOMEDCTCode"].astype(int),
        mapping["Abbreviation"],
    )
)


def load_raw_data(df):
    signals = []

    for file_name in df["FileName"]:
        signal, meta = wfdb.rdsamp(str(Path(DATA_PATH) / file_name))

        if list(meta["sig_name"]) != expected_leads:
            raise ValueError(
                f"Unexpected lead order:\n"
                f"Found    : {meta['sig_name']}\n"
                f"Expected : {expected_leads}"
            )

        if meta["fs"] != 100:
            signal = resample_poly(
                signal,
                up=100,
                down=int(meta["fs"]),
                axis=0,
            )

        if signal.shape[0] > 1000:
            signal = signal[:1000]
        elif signal.shape[0] < 1000:
            pad = np.zeros(
                (1000 - signal.shape[0], signal.shape[1]),
                dtype=np.float32,
            )
            signal = np.vstack((signal, pad))
        signals.append(signal.astype(np.float32))

    return np.asarray(signals)


def build_metadata():
    rows = []

    for hea in sorted(Path(DATA_PATH).rglob("*.hea")):
        record = hea.with_suffix("")
        snomed = ""      

        with open(hea, "r") as f:
            for line in f:
                if line.startswith("# Dx:"):
                    snomed = line.split(":", 1)[1].strip()
                    break

        if snomed == "":
            continue  

        rows.append(
            {
                "FileName": str(record.relative_to(DATA_PATH)),
                "SNOMEDCTCode": snomed,
            }
        )

    return pd.DataFrame(rows)


def aggregate_diagnosis(snomed_string):
    if pd.isna(snomed_string):
        return []
    
    classes = set()
    for diagnosis in str(snomed_string).split(","):        
        try:
            diagnosis = int(diagnosis.strip())
        except ValueError:
            continue

        abbrev = SNOMED_TO_ABBREV.get(diagnosis)

        if abbrev is None:
            continue

        superclass = ABBREV_TO_SUPERCLASS.get(abbrev)

        if superclass is not None:
            classes.add(superclass)

    return sorted(classes)


def load_dataset():
    if os.path.exists(CACHE_X_PATH) and os.path.exists(CACHE_Y_PATH):
        print("Loading cached Chapman dataset...\n")
        X = np.load(CACHE_X_PATH)

        with open(CACHE_Y_PATH, "rb") as f:
            Y = pickle.load(f)

        return X, Y

    print("Building Chapman cache...\n")

    Y = build_metadata()
    before = len(Y)
    Y["diagnostic_superclass"] = Y["SNOMEDCTCode"].apply(aggregate_diagnosis)

    valid = Y["diagnostic_superclass"].apply(len) > 0
    Y = Y.loc[valid].reset_index(drop=True)
    after = len(Y)

    print(f"Original ECGs : {before}")
    print(f"Mapped ECGs   : {after}")
    print(f"Removed ECGs  : {before-after}")
    print()

    X = load_raw_data(Y)
    X = per_lead_external_normalization(X)

    counter = Counter()
    for labels in Y["diagnostic_superclass"]:
        counter.update(labels)

    print("Superclass distribution")

    for cls in ["NORM", "MI", "STTC", "CD", "HYP"]:
        print(f"{cls:5s}: {counter[cls]}")
    print()

    os.makedirs("cache", exist_ok=True)
    np.save(CACHE_X_PATH, X.astype(np.float32))

    with open(CACHE_Y_PATH, "wb") as f:
        pickle.dump(Y, f)

    print("Chapman dataset cached.\n")
    return X, Y


def load_external_validation():
    X, Y = load_dataset()
    return X, Y["diagnostic_superclass"]


def per_lead_external_normalization(X):
    mean = np.load(PTB_MEAN)
    std = np.load(PTB_STD)
    return (X - mean) / std
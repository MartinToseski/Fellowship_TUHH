import pandas as pd
import numpy as np
import wfdb
import ast
import os
import pickle


DATA_PATH = "../../data/ptb-xl/"


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df['filename_lr']]
    else:
        data = [wfdb.rdsamp(path+f) for f in df['filename_hr']]
    data = np.array([signal for signal, meta in data])
    return data


def load_dataset(sampling_rate):
    CACHE_X_PATH = f"../cache/X_{sampling_rate}.npy"
    CACHE_Y_PATH = f"../cache/Y_{sampling_rate}.pkl"

    # ---------- LOAD CACHE IF IT EXISTS ----------
    if os.path.exists(CACHE_X_PATH) and os.path.exists(CACHE_Y_PATH):
        print("Loading cached dataset...", "\n")

        X = np.load(CACHE_X_PATH, allow_pickle=True)

        with open(CACHE_Y_PATH, "rb") as f:
            Y = pickle.load(f)

        return X, Y

    # ---------- BUILD CACHE IF NEEDED ----------
    print("Building dataset cache...", "\n")

    # load and convert annotation data
    Y = pd.read_csv(DATA_PATH+'ptbxl_database.csv', index_col='ecg_id')         # get SCP diagnosis codes 
    Y['scp_codes'] = Y['scp_codes'].apply(lambda x: ast.literal_eval(x))        # keep SCP codes as Python dictionaries

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, DATA_PATH)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(DATA_PATH+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df['diagnostic'] == 1] 
    
    def aggregate_diagnostic(y_dic):
        tmp = []
        #print(y_dic)
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key]['diagnostic_class'])
        #print(list(set(tmp)), "\n")
        return list(set(tmp))                                                   # keep only diagnostic SCP codes

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y['scp_codes'].apply(aggregate_diagnostic)

    # ---------- SAVE CACHE ----------
    os.makedirs("../cache", exist_ok=True)
    np.save(CACHE_X_PATH, X)

    with open(CACHE_Y_PATH, "wb") as f:
        pickle.dump(Y, f)
    
    print("Dataset cached!", "\n")
    return X, Y


def split_data(sampling_rate):
    X, Y = load_dataset(sampling_rate)

    # Split data into train and test
    val_fold = 9
    test_fold = 10

    # Train
    train_mask = ~Y['strat_fold'].isin([val_fold, test_fold])
    X_train = X[train_mask]
    y_train = Y.loc[train_mask, 'diagnostic_superclass']

    # Val
    val_mask = Y['strat_fold'] == val_fold
    X_val = X[val_mask]
    y_val = Y.loc[val_mask, 'diagnostic_superclass']

    # Test
    test_mask = Y['strat_fold'] == test_fold
    X_test = X[test_mask]
    y_test = Y.loc[test_mask, 'diagnostic_superclass']

    return X_train, y_train, X_val, y_val, X_test, y_test


def per_lead_global_normalization(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std = X_train.std(axis=(0,1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test


def per_signal_global_normalization(X_train, X_val, X_test):
    train_mean = X_train.mean(axis=(1, 2), keepdims=True)
    train_std = X_train.std(axis=(1, 2), keepdims=True) + 1e-8

    val_mean = X_val.mean(axis=(1, 2), keepdims=True)
    val_std = X_val.std(axis=(1, 2), keepdims=True) + 1e-8

    test_mean = X_test.mean(axis=(1, 2), keepdims=True)
    test_std = X_test.std(axis=(1, 2), keepdims=True) + 1e-8

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - val_mean) / val_std
    X_test = (X_test - test_mean) / test_std

    return X_train, X_val, X_test


def global_normalization(X_train, X_val, X_test):
    mean = X_train.mean()
    std = X_train.std() + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_val, X_test
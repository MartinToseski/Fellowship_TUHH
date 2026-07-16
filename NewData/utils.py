import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
METRICS = ["train_loss", "val_loss", "train_mae", "val_mae", "train_mse", "val_mse"]


def print_all_sizes(comment, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"{comment} split size...")
    print(f"Train: {len(X_train)}")
    print("Empty diagnosis in y_train:", (y_train.str.len() == 0).sum())
    print(f"Val:   {len(X_val)}")
    print("Empty diagnosis in y_val:", (y_val.str.len() == 0).sum())
    print(f"Test:  {len(X_test)}")
    print("Empty diagnosis in y_test:", (y_test.str.len() == 0).sum())
    print()


def remove_empty_diagnosis(X, Y):
    non_empty_train_mask = Y.str.len() > 0
    X = X[non_empty_train_mask]
    Y = Y[non_empty_train_mask]
    return X, Y


def print_superclass_distribution_statistics(X_train, y_train, X_val, y_val, X_test, y_test):
    total_rows = len(y_train) + len(y_val) + len(y_test)

    total_counts = {}
    for superclass in SUPERCLASSES:
        total_counts[superclass] = sum(superclass in labels for labels in y_train) + sum(superclass in labels for labels in y_val) + sum(superclass in labels for labels in y_test)

    print("Total superclass distribution...")
    for superclass in SUPERCLASSES:
        print(f"{superclass:<6} {total_counts[superclass]:5}   {total_counts[superclass]/total_rows*100:.2f}% of dataset")
    print()

    splits = [
        ("y_train", y_train),
        ("y_val", y_val),
        ("y_test", y_test),
    ]

    for split_name, labels in splits:
        print(split_name)
        for superclass in SUPERCLASSES:
            count = sum(superclass in x for x in labels)
            print(f"{superclass:<6} {count:5}    {count/len(labels)*100:.2f}%   {count/total_counts[superclass]*100:.2f}% of total")
        print()


def plot_metric(df, metric, output_dir):
    metric_df = df[["step", "epoch", metric]].dropna()

    plt.figure()

    if "train" in metric:
        plt.plot(metric_df["step"], metric_df[metric])
        plt.xlabel("Step")
    else:
        plt.plot(metric_df["epoch"], metric_df[metric], marker="o")
        plt.xlabel("Epoch")

    plt.title(metric)

    if "loss" in metric:
        plt.ylabel("Loss")
    elif "auc" in metric:
        plt.ylabel("AUC")
    elif "f1" in metric:
        plt.ylabel("F1 Score")
    elif "precision" in metric:
        plt.ylabel("Precision")
    elif "recall" in metric:
        plt.ylabel("Recall")
    elif "learning_rate" in metric:
        plt.ylabel("Learning Rate")
    else:
        plt.ylabel("Accuracy")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png")
    plt.close()


def plot_all_metrics(log_path):
    log_path = Path(log_path)
    df = pd.read_csv(log_path)
    for metric in METRICS:
        plot_metric(df, metric, log_path.parent)


def print_clean_report(log_path):
    lines = []

    def log(line=""):
        print(line)
        lines.append(line)

    log_path = Path(log_path)
    df = pd.read_csv(log_path)
    latest = df.dropna(subset=["test_acc"]).iloc[-1]

    log("\n" + "="*60)
    log("FINAL TEST REPORT")
    log("="*60)

    log(f"\nOverall:")
    log(f"  Accuracy: {latest['test_acc']:.4f}")
    log(f"  Loss:     {latest['test_loss']:.4f}")

    log("\nPer-class confusion matrix:")
    for i, cls in enumerate(SUPERCLASSES):
        tp = latest.get(f"test_TP_{cls}", np.nan)
        fp = latest.get(f"test_FP_{cls}", np.nan)
        fn = latest.get(f"test_FN_{cls}", np.nan)
        tn = latest.get(f"test_TN_{cls}", np.nan)

        pos_total = tp + fn
        tp_p = (tp / pos_total * 100) if pos_total > 0 else np.nan
        fn_p = (fn / pos_total * 100) if pos_total > 0 else np.nan

        neg_total = fp + tn
        fp_p = (fp / neg_total * 100) if neg_total > 0 else np.nan
        tn_p = (tn / neg_total * 100) if neg_total > 0 else np.nan

        log(f"CONFUSION MATRIX - {cls}")
        log("                                    Predicted")
        log("                            Positive          Negative ")
        log(f"Actual   Positive    {tp:6.0f} ({tp_p:<5.1f}%)   {fn:6.0f} ({fn_p:<5.1f}%)")
        log(f"         Negative    {fp:6.0f} ({fp_p:<5.1f}%)   {tn:6.0f} ({tn_p:<5.1f}%)")
        log()
        log()

    log("Per-class metrics:")
    log(f"Superclass   Precision     Recall       F1 Score        AUC")
    for cls in SUPERCLASSES:
        p = latest.get(f"test_precision_{cls}", np.nan)
        r = latest.get(f"test_recall_{cls}", np.nan)
        f1 = latest.get(f"test_f1_{cls}", np.nan)
        auc = latest.get(f"test_auc_{cls}", np.nan)
        log(f"  {cls:5s} |      {p:.3f}        {r:.3f}         {f1:.3f}        {auc:.3f}")
    log()

    log("Macro:")
    log(f"  F1 macro:  {latest.get('test_f1_macro', np.nan):.4f}")
    log(f"  AUC macro: {latest.get('test_auc_macro', np.nan):.4f}")
    log("="*60 + "\n")

    report_path = log_path.parent / "report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
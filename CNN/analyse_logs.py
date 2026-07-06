from pathlib import Path
import pandas as pd
import yaml

LOG_ROOT = Path("logs/ModernCNN")


def load_experiment(exp_dir: Path, model_name: str):
    metrics_file = exp_dir / "metrics.csv"
    hparams_file = exp_dir / "hparams.yaml"

    if not metrics_file.exists():
        return None

    df = pd.read_csv(metrics_file)

    # Best validation epoch
    val_rows = df[df["val_auc_macro"].notna()].copy()

    if len(val_rows) == 0:
        return None

    best_idx = val_rows["val_auc_macro"].idxmax()
    best = df.loc[best_idx]

    # Final test row
    test_rows = df[df["test_auc_macro"].notna()]

    if len(test_rows) == 0:
        return None

    test = test_rows.iloc[0]

    # Hyperparameters
    hp = {}

    if hparams_file.exists():
        with open(hparams_file, "r") as f:
            hp = yaml.full_load(f)

    return {
        # General
        "model": model_name,
        "experiment": exp_dir.name,
        "folder": str(exp_dir),

        # Hyperparameters
        "learning_rate": hp.get("learning_rate"),
        "batch_size": hp.get("batch_size"),
        "kernel_size": hp.get("kernel_size"),
        "dropout": hp.get("dropout"),
        "weight_decay": hp.get("weight_decay"),
        "optimizer": hp.get("optimizer"),
        "sampling_rate": hp.get("sampling_rate"),

        # Validation
        "best_epoch": int(best["epoch"]),
        "best_val_auc": best["val_auc_macro"],
        "best_val_f1": best["val_f1_macro"],
        "best_val_loss": best["val_loss"],

        # Test
        "test_auc": test["test_auc_macro"],
        "test_f1": test["test_f1_macro"],
        "test_acc": test["test_acc"],
        "test_loss": test["test_loss"],
    }


def main():
    results = []
    model_name = LOG_ROOT.name

    for exp_dir in LOG_ROOT.iterdir():
        if not exp_dir.is_dir():
            continue

        row = load_experiment(exp_dir, model_name)

        if row is not None:
            results.append(row)

    results = pd.DataFrame(results)

    if len(results) == 0:
        print("No experiments found.")
        return

    # Save complete table
    results = results.sort_values(
        ["test_auc", "test_f1"],
        ascending=False
    ).reset_index(drop=True)

    results.to_csv("all_experiments.csv", index=False)

    # ==========================================================
    # Top 10 by TEST AUC
    # ==========================================================
    top_auc = (
        results
        .sort_values("test_auc", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    top_auc.index += 1

    print("\n")
    print("=" * 140)
    print("TOP 10 EXPERIMENTS BY TEST AUC")
    print("=" * 140)

    print(
        top_auc[
            [
                "model",
                "test_auc",
                "test_f1",
                "best_val_auc",
                "best_epoch",
                "learning_rate",
                "batch_size",
                "kernel_size",
                "dropout",
                "weight_decay",
                "optimizer",
                "folder",
            ]
        ].to_string()
    )

    top_auc.to_csv("top10_by_auc.csv", index=False)

    # ==========================================================
    # Top 10 by TEST F1
    # ==========================================================
    top_f1 = (
        results
        .sort_values("test_f1", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    top_f1.index += 1

    print("\n")
    print("=" * 140)
    print("TOP 10 EXPERIMENTS BY TEST F1")
    print("=" * 140)

    print(
        top_f1[
            [
                "model",
                "test_f1",
                "test_auc",
                "best_val_f1",
                "best_epoch",
                "learning_rate",
                "batch_size",
                "kernel_size",
                "dropout",
                "weight_decay",
                "optimizer",
                "folder",
            ]
        ].to_string()
    )

    top_f1.to_csv("top10_by_f1.csv", index=False)

    print("\nSaved:")
    print("  all_experiments.csv")
    print("  top10_by_auc.csv")
    print("  top10_by_f1.csv")


if __name__ == "__main__":
    main()
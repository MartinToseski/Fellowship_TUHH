'''
PREPROCESSING STEPS:
1. WFDB Format Handling                                                     ✓
2. Sampling Rate Handling (100 or 500 Hz)                                   ✓
    + Optional Bandpass Filter                                              -
3. Convert SCP Codes into Binary Vector                                     ✓
4. Signal Per-Record or Per-Lead Normalization                              ✓
5. Class Imbalance Handling - Weighted Loss Function Tested in Grid         -                                                                                      
6. Training/Validation/Split According to Folds                             ✓
    (1-8 training, 9 for validation, and 10 for testing)                    -
7. Reformat signal dimensions depending on model input requirements         ✓
8. Data Augmentation                                                        
'''


import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC, MultilabelF1Score, MultilabelPrecision, MultilabelRecall, MultilabelConfusionMatrix
from torchvision import transforms
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from itertools import product

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from preprocessing import split_data, per_lead_global_normalization
from utils import print_all_sizes, remove_empty_diagnosis, print_superclass_distribution_statistics, plot_all_metrics, print_clean_report


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
pl.seed_everything(22, workers=True)


# ---------- CONFIG DATACLASS ----------
@dataclass
class Config:
    save_dir = "logs"
    model_name: str = "ModernCNN"

    sampling_rate: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.2
    kernel_size: int = 3
    
    augmentation: str = "both"
    optimizer: str = "adam"

    num_classes: int = 5
    max_epochs: int = 3
    threshold: float = 0.5


# ---------- LIGHTNING DATASET ----------
class ECGDataset(Dataset):
    def __init__(self, X, y, augmentation=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augmentation == "time_shift":
            shift = np.random.randint(-50, 51)
            x = torch.roll(x, shifts=shift, dims=1)
        elif self.augmentation == "gaussian_noise":
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        elif self.augmentation == "both":
            shift = np.random.randint(-50, 51)
            x = torch.roll(x, shifts=shift, dims=1)
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        return x, y


# ---------- LIGHTNING DATA MODULE ----------
class ECGDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(self.config.sampling_rate)
        #print_all_sizes("Initial", X_train, y_train, X_val, y_val, X_test, y_test)

        # remove epty labels
        X_train, y_train = remove_empty_diagnosis(X_train, y_train)
        X_val, y_val = remove_empty_diagnosis(X_val, y_val)
        X_test, y_test = remove_empty_diagnosis(X_test, y_test)
        #print_all_sizes("After removing", X_train, y_train, X_val, y_val, X_test, y_test)
        #print_superclass_distribution_statistics(X_train, y_train, X_val, y_val, X_test, y_test)

        # label encoding
        mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
        y_train = mlb.fit_transform(y_train)
        y_val = mlb.transform(y_val)
        y_test = mlb.transform(y_test)

        # normalization
        X_train, X_val, X_test = per_lead_global_normalization(X_train, X_val, X_test)

        # format for Conv1D (batch, channels, time)
        X_train = np.transpose(X_train, (0, 2, 1))
        X_val = np.transpose(X_val, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

        self.train_dataset = ECGDataset(X_train, y_train, augmentation=self.config.augmentation)
        self.val_dataset = ECGDataset(X_val, y_val)
        self.test_dataset = ECGDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )


# ---------- LIGHTNING MODULE ----------
class ECGLitModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_name = config.model_name

        self.val_probs = []
        self.val_targets = []

        self.test_probs = []
        self.test_targets = []
        
        # ---------------- Modern CNN ----------------
        k = config.kernel_size
        p = k // 2

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(12, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, kernel_size=k, padding=p),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=k, padding=p),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=k, padding=p),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=k, padding=p),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, kernel_size=k, padding=p),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2),

            # Block 4
            nn.Conv1d(256, 512, kernel_size=k, padding=p),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 512, kernel_size=k, padding=p),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(512, config.num_classes)

        # Multi-label safe metrics
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_acc = MultilabelAccuracy(num_labels=5, threshold=config.threshold)

        self.val_acc = MultilabelAccuracy(num_labels=5, threshold=config.threshold)
        self.val_auc = MultilabelAUROC(num_labels=5, average=None)
        self.val_auc_macro = MultilabelAUROC(num_labels=5, average="macro")
        self.val_f1 = MultilabelF1Score(num_labels=5, average=None, threshold=config.threshold)
        self.val_f1_macro = MultilabelF1Score(num_labels=5, average="macro", threshold=config.threshold)
        self.val_precision = MultilabelPrecision(num_labels=5, average=None, threshold=config.threshold)
        self.val_recall = MultilabelRecall(num_labels=5, average=None, threshold=config.threshold)

        self.test_acc = MultilabelAccuracy(num_labels=5, threshold=config.threshold)
        self.test_auc = MultilabelAUROC(num_labels=5, average=None)
        self.test_auc_macro = MultilabelAUROC(num_labels=5, average="macro")
        self.test_f1_macro = MultilabelF1Score(num_labels=5, average="macro", threshold=config.threshold)

        self.test_f1 = MultilabelF1Score(
            num_labels=5,
            average=None,
            threshold=self.config.threshold
        )

        self.test_precision = MultilabelPrecision(
            num_labels=5,
            average=None,
            threshold=self.config.threshold
        )

        self.test_recall = MultilabelRecall(
            num_labels=5,
            average=None,
            threshold=self.config.threshold
        )
        
        self.test_cm = MultilabelConfusionMatrix(num_labels=5, threshold=config.threshold)
        self.val_cm = MultilabelConfusionMatrix(num_labels=5, threshold=config.threshold)

        # Hyperparameters logging
        self.save_hyperparameters(vars(config))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)

        loss = self.loss_fn(logits, y)
        self.train_acc(probs, y.int())

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        
        loss = self.loss_fn(logits, y)
        self.val_acc(probs, y.int())
        self.val_auc(probs, y.int())
        self.val_auc_macro(probs, y.int())
        self.val_f1(probs, y.int())
        self.val_f1_macro(probs, y.int())
        self.val_precision(probs, y.int())
        self.val_recall(probs, y.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)

        loss = self.loss_fn(logits, y)

        self.test_acc(probs, y.int())
        self.test_auc(probs, y.int())
        self.test_auc_macro(probs, y.int())
        self.test_f1(probs, y.int())
        self.test_f1_macro(probs, y.int())
        self.test_precision(probs, y.int())
        self.test_recall(probs, y.int())

        # confusion matrix can use preds
        preds = (probs >= self.config.threshold).int()
        self.test_cm(preds, y.int())

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_acc.reset()
        self.test_auc.reset()
        self.test_auc_macro.reset()
        self.test_f1.reset()
        self.test_f1_macro.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_cm.reset()
        
    def on_validation_epoch_end(self):
        for i, cls in enumerate(SUPERCLASSES):
            self.log(f"val_auc_{cls}", self.val_auc.compute()[i])
            self.log(f"val_f1_{cls}", self.val_f1.compute()[i])
            self.log(f"val_precision_{cls}", self.val_precision.compute()[i])
            self.log(f"val_recall_{cls}", self.val_recall.compute()[i])

        self.log("val_auc_macro", self.val_auc_macro.compute(), prog_bar=True)
        self.log("val_f1_macro", self.val_f1_macro.compute(), prog_bar=True)

        self.val_auc.reset()
        self.val_auc_macro.reset()
        self.val_f1.reset()
        self.val_f1_macro.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def on_test_epoch_end(self):
        auc = self.test_auc.compute()
        f1 = self.test_f1.compute()
        prec = self.test_precision.compute()
        rec = self.test_recall.compute()
        
        cm = self.test_cm.compute()  # shape: [classes, 2, 2]
        for i, cls in enumerate(SUPERCLASSES):
            tn = cm[i, 0, 0].item()
            fp = cm[i, 0, 1].item()
            fn = cm[i, 1, 0].item()
            tp = cm[i, 1, 1].item()

            self.log(f"test_TP_{cls}", tp)
            self.log(f"test_FP_{cls}", fp)
            self.log(f"test_FN_{cls}", fn)
            self.log(f"test_TN_{cls}", tn)

        for i, cls in enumerate(SUPERCLASSES):
            self.log(f"test_auc_{cls}", auc[i])
            self.log(f"test_f1_{cls}", f1[i])
            self.log(f"test_precision_{cls}", prec[i])
            self.log(f"test_recall_{cls}", rec[i])

        self.log("test_auc_macro", self.test_auc_macro.compute(), prog_bar=True)
        self.log("test_f1_macro", self.test_f1_macro.compute(), prog_bar=True)

        self.test_auc.reset()
        self.test_auc_macro.reset()
        self.test_f1.reset()
        self.test_f1_macro.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_cm.reset()

    def configure_optimizers(self):
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )


# ---------- LIGHTNING TRAINER ----------
def run_experiment(config):
    model = ECGLitModule(config)
    data = ECGDataModule(config)

    version = (
        f"ks{config.kernel_size}"
        f"_dr{config.dropout}"
        f"_lr{config.learning_rate}"
        f"_rate{config.sampling_rate}"
        f"_epochs{config.max_epochs}"
        f"_{config.optimizer}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger = CSVLogger(save_dir=config.save_dir, name=config.model_name, version=version)
    checkpoint = ModelCheckpoint(monitor="val_auc_macro", mode="max", save_top_k=1, filename="{epoch}-{val_auc_macro:.4f}")
    early_stop = EarlyStopping(monitor="val_auc_macro", mode="max", patience=5, min_delta=0.001, verbose=True)

    trainer = pl.Trainer(max_epochs=config.max_epochs, logger=logger, callbacks=[checkpoint, early_stop], devices=1)
    trainer.fit(model, datamodule=data)
    trainer.test(model=model, datamodule=data, ckpt_path=checkpoint.best_model_path, verbose=False)

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    return metrics_path


# ---------- GRID SEARCH ----------
grid = {
    "learning_rate": [1e-3, 1e-4],
    "batch_size": [64, 256],
    "dropout": [0.3, 0.5],
    "kernel_size": [3, 5, 7],
    "weight_decay": [0.0, 1e-4]
}

keys = grid.keys()
values = grid.values()
combinations = list(product(*values))

results = []

for combo in combinations:
    params = dict(zip(keys, combo))

    config = Config(
        model_name="ModernCNN",
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        dropout=params["dropout"],
        kernel_size=params["kernel_size"],
        weight_decay=params["weight_decay"],
        max_epochs=50
    )

    print("=" * 80)
    print("!!!")
    print("NEW CONFIG:")
    print("!!!")
    print(config)
    print()
    print()

    metrics_path = run_experiment(config)
    plot_all_metrics(metrics_path)
    print_clean_report(metrics_path)

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

from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MultilabelAccuracy
from torchvision import transforms
from dataclasses import dataclass

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from preprocessing import split_data, per_lead_global_normalization
from utils import print_all_sizes, remove_empty_diagnosis, print_superclass_distribution_statistics


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
pl.seed_everything(22, workers=True)


# ---------- CONFIG DATACLASS ----------
@dataclass
class Config:
    save_dir = "logs"
    model_name: str = "LeNet-5"

    sampling_rate: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    augmentation: str = "time_shift"
    optimizer: str = "adam"

    num_classes: int = 5
    max_epochs: int = 3
    threshold: float = 0.5


# ---------- LIGHTNING DATASET ----------
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------- LIGHTNING DATA MODULE ----------
class ECGDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(self.config.sampling_rate)
        print_all_sizes("Initial", X_train, y_train, X_val, y_val, X_test, y_test)

        # remove epty labels
        X_train, y_train = remove_empty_diagnosis(X_train, y_train)
        X_val, y_val = remove_empty_diagnosis(X_val, y_val)
        X_test, y_test = remove_empty_diagnosis(X_test, y_test)
        print_all_sizes("After removing", X_train, y_train, X_val, y_val, X_test, y_test)
        print_superclass_distribution_statistics(X_train, y_train, X_val, y_val, X_test, y_test)

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

        self.train_dataset = ECGDataset(X_train, y_train)
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
LENET_ARCH = [
    (32, 7, 3),   # conv1
    (64, 5, 2),   # conv2
    (128, 3, 1),  # conv3
]

class ECGLitModule(pl.LightningModule):
    def __init__(self, config: Config, architecture):
        super().__init__()
        self.config = config
        self.architecture = architecture
        self.model_name = config.model_name
        
        # Build CNN
        layers = []
        in_ch = 12
        for out_ch, k, p in architecture:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ]
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)

        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, config.num_classes)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # Multi-label safe metrics
        self.train_acc = MultilabelAccuracy(num_labels=config.num_classes, threshold=config.threshold)
        self.val_acc = MultilabelAccuracy(num_labels=config.num_classes, threshold=config.threshold)
        self.test_acc = MultilabelAccuracy(num_labels=config.num_classes, threshold=config.threshold)

        # Architecture logging
        self.save_hyperparameters(ignore=["architecture", "config"])

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        acc = self.train_acc(preds, y.int())

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        acc = self.val_acc(preds, y.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        acc = self.test_acc(preds, y.int())

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        
    def configure_optimizers(self):
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")


# ---------- LIGHTNING TRAINER ----------
config = Config(model_name="LeNet")
model = ECGLitModule(config, LENET_ARCH)
data = ECGDataModule(config)

logger = CSVLogger(save_dir=config.save_dir, name=config.model_name)
checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="{epoch}-{val_loss:.4f}")

trainer = pl.Trainer(max_epochs=config.max_epochs, logger=logger, callbacks=[checkpoint], devices=1)
trainer.fit(model, datamodule=data)
trainer.test(model=model, datamodule=data, ckpt_path=checkpoint.best_model_path)


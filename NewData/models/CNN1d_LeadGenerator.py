import sys
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError 
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessing import split_data, per_lead_global_normalization
from utils.utils import plot_all_metrics


pl.seed_everything(22, workers=True)


# ---------- CONFIG DATACLASS ----------
@dataclass
class Config:
    save_dir = "./logs"
    model_name: str = "ModernCNN"
    sampling_rate: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.2
    optimizer: str = "adam"
    max_epochs: int = 3


# ---------- LIGHTNING DATASET ----------
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].clone(), self.y[idx]


# ---------- LIGHTNING DATA MODULE ----------
class ECGDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(self.config.sampling_rate)

        # normalization
        X_train, X_val, X_test = per_lead_global_normalization(X_train, X_val, X_test)

        Y_train = X_train[:, :, :6].copy()
        Y_val   = X_val[:, :, :6].copy()
        Y_test  = X_test[:, :, :6].copy()

        X_train = X_train[:, :, 6:12]
        X_val = X_val[:, :, 6:12]
        X_test = X_test[:, :, 6:12]

        # format for Conv1D (batch, channels, time)
        X_train = np.transpose(X_train, (0, 2, 1))
        X_val = np.transpose(X_val, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

        Y_train = np.transpose(Y_train, (0, 2, 1))
        Y_val = np.transpose(Y_val, (0, 2, 1))
        Y_test = np.transpose(Y_test, (0, 2, 1))

        self.train_dataset = ECGDataset(X_train, Y_train)
        self.val_dataset   = ECGDataset(X_val, Y_val)
        self.test_dataset  = ECGDataset(X_test, Y_test)

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
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, bias=False)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + identity
        x = self.act(x)

        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels)
        self.down = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.res(x)
        skip = x
        x = self.down(x)
        return x, skip
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.size(-1) != skip.size(-1):
            x = F.interpolate(
                x,
                size=skip.size(-1),
                mode="linear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        x = self.res(x)

        return x
    

class ECGLitModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        
        # ---------------- Modern CNN ----------------
        # Encoder
        self.stem = nn.Conv1d(6, 64, kernel_size=7, padding=3)

        self.enc1 = EncoderBlock(64, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )

        # Decoder
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        # Output
        self.head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 6, kernel_size=1),
        )

        # Multi-label safe metrics
        self.loss_fn = nn.SmoothL1Loss()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        # Hyperparameters logging
        self.save_hyperparameters(vars(config))

    def forward(self, x):
        x = self.stem(x)

        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)

        x = self.bottleneck(x)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        x = self.head(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.train_mae(pred, y)
        self.train_mse(pred, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mae", self.train_mae)
        self.log("train_mse", self.train_mse)

        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.val_mae(pred, y)
        self.val_mse(pred, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_mae)
        self.log("val_mse", self.val_mse)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.test_mae(pred, y)
        self.test_mse(pred, y)

        self.log("test_loss", loss)
        self.log("test_mae", self.test_mae)
        self.log("test_mse", self.test_mse)

        return loss

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
        f"dr{config.dropout}"
        f"_lr{config.learning_rate}"
        f"_rate{config.sampling_rate}"
        f"_epochs{config.max_epochs}"
        f"_{config.optimizer}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    wandb_logger = WandbLogger(project="ECG-Lead-Reconstruction", name=version, log_model=True)
    logger = CSVLogger(save_dir=config.save_dir, name=config.model_name, version=version)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="{epoch}-{val_loss:.4f}")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=15, min_delta=0.0001, verbose=True)

    trainer = pl.Trainer(max_epochs=config.max_epochs, logger=[logger, wandb_logger], callbacks=[checkpoint, early_stop], devices=[1])
    trainer.fit(model, datamodule=data)
    trainer.test(model=model, datamodule=data, ckpt_path=checkpoint.best_model_path, verbose=False)
    
    wandb_logger.experiment.summary["best_checkpoint"] = checkpoint.best_model_path
    wandb_logger.experiment.summary["best_val_loss"] = checkpoint.best_model_score.item()

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    return metrics_path


# ---------- GRID SEARCH ----------
if __name__ == "__main__":    
    for i in range(1):
        config = Config(
            model_name="Lead_Reconstruction_CNN",

            sampling_rate=100,

            batch_size=256,
            learning_rate=1e-3,
            weight_decay=0.0,
            dropout=0.3,

            optimizer="adam",
            max_epochs=100,
        )

        metrics_path = run_experiment(config)
        plot_all_metrics(metrics_path)

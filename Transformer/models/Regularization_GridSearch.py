import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import math
import sys
import wandb

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC, MultilabelF1Score, MultilabelPrecision, MultilabelRecall, MultilabelConfusionMatrix
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
from itertools import product

from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


from utils.preprocessing import split_data, per_lead_global_normalization, per_signal_global_normalization, global_normalization
from utils.utils import print_all_sizes, remove_empty_diagnosis, print_superclass_distribution_statistics, plot_all_metrics, print_clean_report


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
pl.seed_everything(22, workers=True)


# ---------- CONFIG DATACLASS ----------
@dataclass
class Config:
    save_dir = "../logs"
    model_name: str = "Transformer"

    sampling_rate: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    
    dropout: float = 0.2
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_dim: int = 512

    patch_size: int = 5
    pooling: str = "cls"

    positional_encoding: str = "sinusoidal"
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    augmentation: str = "both"
    optimizer: str = "adamw"

    num_classes: int = 5
    max_epochs: int = 50
    threshold: float = 0.5

    patience: int = 10
    early_stop_threshold: float = 1e-4
    gradient_clip_val: float = 1.0


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

        # weights for loss function
        positive = y_train.sum(axis=0)
        negative = len(y_train) - positive
        pos_weight = np.sqrt(negative / positive)
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        # normalization
        X_train, X_val, X_test = per_lead_global_normalization(X_train, X_val, X_test)

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


# ---------- TRANSFORMER MODULES ----------
class PatchEmbedding(nn.Module):
    '''
        Input:
            (batch, sequence_length, num_leads) / 
            (batch, 1000, 12)
        Output:
            (batch, num_patches, d_model)
            (batch, 200, d_model)
    '''
    def __init__(self, patch_size, num_leads, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.num_leads = num_leads   
        self.projection = nn.Linear(patch_size*num_leads, d_model)

    def forward(self, x):
        B, T, C = x.shape

        if T % self.patch_size != 0:
            raise ValueError(
                f"Sequence length ({T}) must be divisible "
                f"by patch_size ({self.patch_size})."
            )

        num_patches = T // self.patch_size

        x = x.reshape(B, num_patches, self.patch_size*C)
        x = self.projection(x)
        return x


class PositionalEncoding(nn.Module):
    '''
        PE(pos,2i) = sin(pos/10000^(2i/d_model))
        PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    '''
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ---------- LIGHTNING MODULE ----------
class ECGLitModule(pl.LightningModule):
    def __init__(self, config: Config, pos_weight=None):
        super().__init__()
        self.config = config
        self.model_name = config.model_name

        self.val_probs = []
        self.val_targets = []

        self.test_probs = []
        self.test_targets = []

        assert config.d_model % config.n_heads == 0
        
        # ---------------- Transformer Architecture ----------------
        self.patch_embedding = PatchEmbedding(
            patch_size=config.patch_size,
            num_leads=12,
            d_model=config.d_model,
        )

        self.embedding_dropout = nn.Dropout(config.dropout)

        num_patches = config.sampling_rate*10 // config.patch_size

        if config.positional_encoding == "sinusoidal":
            self.positional_encoding = PositionalEncoding(
                d_model=config.d_model,
                max_len=num_patches + 1,
            )
        elif config.positional_encoding == "learnable":
            self.positional_embedding = nn.Parameter(
                torch.randn(1, num_patches + 1, config.d_model)
            )

            nn.init.trunc_normal_(
                self.positional_embedding,
                std=0.02
            )    

        self.cls_token = nn.Parameter(
            torch.empty(1,1,config.d_model)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )

        self.dropout = nn.Dropout(config.dropout)

        self.norm = nn.LayerNorm(config.d_model)

        self.fc = nn.Linear(
            config.d_model,
            config.num_classes
        )

        # Multi-label safe metrics
        if pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
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
        x = self.patch_embedding(x)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        if self.config.positional_encoding == "sinusoidal":
            x = self.positional_encoding(x)
        else:
            x = x + self.positional_embedding

        x = self.embedding_dropout(x)
        x = self.encoder(x)

        if self.config.pooling == "cls":
            x = x[:,0]
        elif self.config.pooling == "mean":
            x = x.mean(dim=1)

        x = self.norm(x)
        x = self.dropout(x)

        logits = self.fc(x)

        return logits

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

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_acc.reset()
        self.test_auc.reset()
        self.test_auc_macro.reset()
        self.test_f1.reset()
        self.test_f1_macro.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_cm.reset()

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            prog_bar=False,
            logger=True,
            on_epoch=True,
        )
        
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
        
        cm = self.test_cm.compute()  
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )   

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_epochs - self.config.warmup_epochs,
            eta_min=self.config.min_lr,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                cosine_scheduler,
            ],
            milestones=[
                self.config.warmup_epochs
            ],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ---------- LIGHTNING TRAINER ----------
def run_experiment(config):
    data = ECGDataModule(config)
    data.setup()

    model = ECGLitModule(config, pos_weight=data.pos_weight)

    version = (
        f"d{config.d_model}"
        f"_head{config.n_heads}"
        f"_lay{config.n_layers}"
        f"_ff{config.ff_dim}"
        f"_ptch{config.patch_size}"
        f"_pl{config.pooling}"
        f"_pos{config.positional_encoding}"
        f"_dr{config.dropout}"
        f"_lr{config.learning_rate}"
        f"_ep{config.max_epochs}"
        f"_opt{config.optimizer}"
        f"_pat{config.patience}"
        f"_patt{config.early_stop_threshold}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger = CSVLogger(save_dir=config.save_dir, name=config.model_name, version=version)
    wandb_logger = WandbLogger(project="Regularization_GridSearch", entity="martintoseski13-kaunas-university-of-technology", name=version, log_model=True)

    wandb_logger.log_hyperparams(vars(config))
    num_params = sum(p.numel() for p in model.parameters())
    wandb_logger.experiment.config.update({"parameters": num_params})

    wandb_logger.watch(model, log="gradients", log_freq=100)

    checkpoint = ModelCheckpoint(monitor="val_f1_macro", mode="max", save_top_k=1, filename="{epoch:02d}-{val_f1_macro:.4f}-{val_auc_macro:.4f}")
    early_stop = EarlyStopping(monitor="val_f1_macro", mode="max", patience=config.patience, min_delta=config.early_stop_threshold, verbose=True)

    trainer = pl.Trainer(max_epochs=config.max_epochs, logger=[logger, wandb_logger], callbacks=[checkpoint, early_stop], gradient_clip_val=config.gradient_clip_val, devices=[0])
    trainer.fit(model, datamodule=data)
    trainer.test(model=model, datamodule=data, ckpt_path=checkpoint.best_model_path, verbose=False)

    wandb_logger.experiment.summary["best_checkpoint"] = checkpoint.best_model_path
    wandb_logger.experiment.summary["best_val_f1"] = checkpoint.best_model_score.item()

    # Validation AUC stored inside checkpoint callback filename/metrics
    if "val_auc_macro" in trainer.callback_metrics:
        wandb_logger.experiment.summary["best_val_auc"] = (
            trainer.callback_metrics["val_auc_macro"].item()
        )

    wandb_logger.experiment.summary["parameters"] = num_params
    wandb.finish()

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    return metrics_path


# ---------- GRID SEARCH ----------
if __name__ == "__main__":
    grid = [
        # Baseline
        {
            "learning_rate": 3e-4,
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "patch_size": 5,
        },

        # ---------- Learning Rate ----------
        {
            "learning_rate": 1e-4,
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "patch_size": 5,
        },

        {
            "learning_rate": 5e-4,
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "patch_size": 5,
        },

        {
            "learning_rate": 1e-3,
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "patch_size": 5,
        },

        # ---------- Patch Size ----------
        {
            "learning_rate": 3e-4,
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "patch_size": 4,
        },

        {
            "learning_rate": 3e-4,
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "patch_size": 10,
        },

        {
            "learning_rate": 3e-4,
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "patch_size": 20,
        },

        # ---------- Dropout ----------
        {
            "learning_rate": 3e-4,
            "dropout": 0.1,
            "weight_decay": 1e-2,
            "patch_size": 5,
        },

        {
            "learning_rate": 3e-4,
            "dropout": 0.3,
            "weight_decay": 1e-2,
            "patch_size": 5,
        },

        {
            "learning_rate": 3e-4,
            "dropout": 0.4,
            "weight_decay": 1e-2,
            "patch_size": 5,
        },

        # ---------- Weight Decay ----------
        {
            "learning_rate": 3e-4,
            "dropout": 0.2,
            "weight_decay": 0.0,
            "patch_size": 5,
        },

        {
            "learning_rate": 3e-4,
            "dropout": 0.2,
            "weight_decay": 1e-3,
            "patch_size": 5,
        },

        {
            "learning_rate": 3e-4,
            "dropout": 0.2,
            "weight_decay": 5e-2,
            "patch_size": 5,
        },
    ]

    results = []
    for params in grid:
        config = Config(
            model_name="Regularization_GridSearch",

            sampling_rate=100,
            augmentation="both",

            batch_size=64,
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            dropout=params["dropout"],

            d_model = 384,
            n_heads = 8,
            n_layers = 6,
            ff_dim = 2304,

            patch_size = params["patch_size"],
            pooling = "cls",

            positional_encoding = "sinusoidal",

            num_classes=5,
            max_epochs=100,
            warmup_epochs=10,
            threshold=0.5,

            patience=15,
            early_stop_threshold=1e-4,
            gradient_clip_val=1.0
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

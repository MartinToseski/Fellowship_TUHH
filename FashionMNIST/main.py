import os 
from dataclasses import dataclass
import pandas as pd
import numpy as np

from typing import Tuple

import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


pl.seed_everything(22, workers=True)


@dataclass
class Config:
    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    save_dir: str = "logs/"
    batch_size: int = 256 if torch.cuda.is_available() else 64
    max_epochs: int = 20
    accelerator: str = "auto"
    devices: int = 1

config = Config()


class LitFashionMNIST(pl.LightningModule):
    def __init__(self, data_dir: str = config.data_dir, batch_size: int = config.batch_size, learning_rate = 0.01, model_name="cnn_v1"):
        super().__init__()

        self.save_hyperparameters()

        self.num_classes = 10
        self.dims = (1, 28, 28)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(28, padding=2),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )

        self.model = nn.Sequential(                                                             # 1 x 28 x 28
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),      # 32 x 28 x 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),     # 32 x 28 x 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                              # 32 x 14 x 14
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),     # 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),     # 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                              # 64 x 7 x 7
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),    # 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),   # 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                              # 128 x 3 x 3

            #nn.AdaptiveAvgPool2d((1, 1)),                                                       # 128 x 1 x 1
            
            nn.Flatten(),
            
            nn.Linear(in_features=128*3*3, out_features=120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=120, out_features=10),
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.example_input_array = torch.randn(1, *self.dims)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)

        preds = torch.argmax(output, dim=1)
        self.val_accuracy(preds, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def test_step(self, batch, batch_nb):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)

        preds = torch.argmax(output, dim=1)
        self.test_accuracy(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def prepare_data(self):
        FashionMNIST(self.hparams.data_dir, train=True, download=True)
        FashionMNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage):
        if stage == "fit" or stage is None:
            fashionMNIST = FashionMNIST(self.hparams.data_dir, train=True, transform=self.transform)
            generator = torch.Generator().manual_seed(22)
            self.fmnist_train, self.fmnist_val = random_split(fashionMNIST, [55000, 5000], generator=generator)

        if stage == "test" or stage is None:
            self.fmnist_test = FashionMNIST(self.hparams.data_dir, train=False, transform=self.transform)
            
    def train_dataloader(self):
        return DataLoader(self.fmnist_train, batch_size=self.hparams.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.fmnist_val, batch_size=self.hparams.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.fmnist_test, batch_size=self.hparams.batch_size)
    

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_top_k=1,
    filename="{epoch}-{val_acc:.4f}"
)


model = LitFashionMNIST(learning_rate=0.001, model_name="LeNet5_v1")

num_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
print(f"Trainable parameters: {num_params:,}")

trainer = pl.Trainer(accelerator=config.accelerator, devices=config.devices, max_epochs=config.max_epochs, logger=CSVLogger(save_dir=config.save_dir, name=model.hparams.model_name), profiler="simple", callbacks=[checkpoint_callback])
trainer.fit(model)
trainer.test(ckpt_path="best")
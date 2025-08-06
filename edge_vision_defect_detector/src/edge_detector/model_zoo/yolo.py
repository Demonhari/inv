from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from ultralytics import YOLO


class YOLOv8Module(pl.LightningModule):
    """
    Lightning wrapper so YOLOv8 plays nicely with the Trainer, MLflow autologging,
    and our RL sampling loop.
    """

    def __init__(self, model_name: str = "yolov8n.pt", lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.yolo = YOLO(model_name)
        self.loss_fn = self.yolo.model.loss
        self.example_input_array = torch.randn(1, 3, 640, 640)

    def forward(self, x):
        return self.yolo(x, verbose=False)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self.yolo.model(imgs)
        loss = self.loss_fn(preds, targets)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.yolo.model.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        return self(batch)

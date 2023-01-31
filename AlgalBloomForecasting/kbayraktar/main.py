import importlib
import wandb
import os
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from typing import Optional

# Increase the timeout duration of the _wait_for_ports function from 30 seconds to 300 seconds.
# This patch fixes wandb failing to find ports on a slow cluster. not as relevant if remote processors are not used.
if "SLURM_JOB_ID" in os.environ:
    import subprocess
    import wandb.sdk.service.service  

    def _wait_for_ports_decorator(original_method):
        def _wait_for_ports(self, fname: str, proc: Optional[subprocess.Popen] = None) -> bool:
            return any(original_method(self,fname, proc) for _ in range(10))
        return _wait_for_ports
    
    wandb.sdk.service.service._Service._wait_for_ports = \
        _wait_for_ports_decorator(wandb.sdk.service.service._Service._wait_for_ports)

unet = importlib.import_module("unet.unet_model")

if __name__ == "__main__":
    #Set seed
    torch.manual_seed(42)

    # Define the loggers.
    wandb_logger = WandbLogger(project="unet-classification", log_model=True)

    # Define the different trainers.
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=20,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        detect_anomaly=True,
        deterministic="warn",
    )

    #U-Net as classification task
    #initialize the model specifications
    unet_classification = unet.UNet(
        root='/../', # to be filled in
        reservoir='palmar',
        window_size=1,
        n_bands=12, # +1 for time as a band
        n_classes=5,
        learning_rate=1e-4,
        weight_decay=1e-5, 
        train_samples=1000,
        batch_size=16,
        num_workers=8,
    ).float()

    # Train, evaluate and test model.
    trainer.fit(unet_classification)

    # used for data analysis purposes
    predictor = Trainer(accelerator='gpu', devices=1,detect_anomaly=True)
    predictions = predictor.predict(unet_classification)  
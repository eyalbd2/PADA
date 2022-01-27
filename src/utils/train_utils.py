"""
Author: Nadav Oved (@nadavo, nadavo@gmail.com), 2021.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from multiprocessing import cpu_count
import numpy as np
import torch as pt
import logging
import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)
    if pt.cuda.is_available():
        pt.cuda.manual_seed_all(seed)
    seed_everything(seed)


def count_num_cpu_gpu() -> Tuple[int, int]:
    num_cpu_cores = cpu_count() // 2
    if pt.cuda.is_available():
        num_gpu_cores = pt.cuda.device_count()
        num_cpu_cores = num_cpu_cores // num_gpu_cores
    else:
        num_gpu_cores = 0
    return num_cpu_cores, num_gpu_cores


NUM_CPU, NUM_GPU = count_num_cpu_gpu()


class LoggingCallback(Callback):
    def __init__(self, logger: logging.Logger = None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using GPU: {pt.cuda.is_available()}")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        self.logger.info("***** Validation results *****")
        metrics = filter(lambda x: x[0] not in ["log", "progress_bar"], trainer.callback_metrics.items())
        # Log results
        for key, metric in sorted(metrics):
            self.logger.info(f"{key} = {metric:.03f}\n")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        self.logger.info("***** Test results *****")
        self.logger.info(f"Num Training Epochs: {trainer.max_epochs}")
        # Log and save results to file
        output_dir = Path(trainer.default_root_dir)
        pl_module.write_eval_predictions(output_dir, "test")
        metrics = filter(lambda x: x[0] not in ["log", "progress_bar"], trainer.callback_metrics.items())
        with open(output_dir / "test_results.txt", "w") as writer:
            for key, metric in sorted(metrics):
                self.logger.info(f"{key} = {metric:.03f}\n")
                writer.write(f"{key} = {metric:.03f}\n")


class ModelCheckpointWithResults(ModelCheckpoint):
    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule) -> Dict[str, Any]:
        pl_module.write_eval_predictions(Path(trainer.default_root_dir), "dev")
        return super().on_save_checkpoint(trainer, pl_module)

    ### For Pytorch-lightning versions > 1.2.1 - use this instead
    # def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> Dict[
    #         str, Any]:
    #      pl_module.write_eval_predictions(Path(trainer.default_root_dir), "dev")
    #      return super().on_save_checkpoint(trainer, pl_module, checkpoint)
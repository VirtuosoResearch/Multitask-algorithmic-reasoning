import os
import sys
from typing import Optional, Any, Dict
from collections import defaultdict
import csv

import torch
from loguru import logger
import lightning.pytorch as pl
import argparse
import os
import math

from core.module import SALSACLRSModel
from core.config import load_cfg
from core.utils import NaNException
from data_loader import CLRSData, CLRSDataset, CLRSDataModule

logger.remove()
logger.add(sys.stderr, level="INFO")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train(model, datamodule, cfg, specs, seed=42, checkpoint_dir=None):
    callbacks = []
    # checkpointing
    if checkpoint_dir is not None:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join("./checkpoints", str(cfg.ALGORITHM), cfg.RUN_NAME), 
            monitor="val_loss", mode="min", filename=f'seed{seed}-{{epoch}}-{{step}}', save_top_k=1, verbose=True)
        callbacks.append(ckpt_cbk)

    # early stopping
    early_stop_cbk = pl.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.TRAIN.EARLY_STOPPING_PATIENCE, mode="min", verbose=True)
    callbacks.append(early_stop_cbk)

    # Setup trainer
    trainer = pl.Trainer(
        devices=[0],
        enable_checkpointing=True,
        callbacks=[ckpt_cbk, early_stop_cbk],
        max_epochs=cfg.TRAIN.MAX_EPOCHS,
        logger=None,
        accelerator="auto",
        log_every_n_steps=5,
        gradient_clip_val=cfg.TRAIN.GRADIENT_CLIP_VAL,
        reload_dataloaders_every_n_epochs=datamodule.reload_every_n_epochs,
        precision= cfg.TRAIN.PRECISION,
    )

    # Load checkpoint
    if cfg.TRAIN.LOAD_CHECKPOINT is not None:
        logger.info(f"Loading checkpoint from {cfg.TRAIN.LOAD_CHECKPOINT}")
        model = SALSACLRSModel.load_from_checkpoint(cfg.TRAIN.LOAD_CHECKPOINT, cfg=cfg, specs=specs)

    # Train
    if cfg.TRAIN.ENABLE:
        try:
            logger.info("Starting training...")
            trainer.fit(model, datamodule=datamodule)
        except NaNException:
            logger.info(f"NaN detected, trying to recover from {ckpt_cbk.best_model_path}...")
            try:
                trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_cbk.best_model_path)
            except NaNException:
                logger.info("Recovery failed, stopping training...")

    # Load best model
    if cfg.TRAIN.LOAD_CHECKPOINT is None and cfg.TRAIN.ENABLE:
        logger.info(f"Best model path: {ckpt_cbk.best_model_path}")
        model = SALSACLRSModel.load_from_checkpoint(ckpt_cbk.best_model_path)

    # Test
    logger.info("Testing best model...")
    results = trainer.test(model, datamodule=datamodule)
    print(results)

    # Log results
    stacked_results = {}
    for d in results:
        stacked_results.update(d)

    logger.info(stacked_results)
    logger.info("Saving results...")
    results_dir = f"results/{cfg.ALGORITHM}/{cfg.RUN_NAME}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # write csv
    with open(os.path.join(results_dir, f"{seed}.csv"), "w") as f:
        writer = csv.DictWriter(f, stacked_results.keys())
        writer.writeheader()
        writer.writerow(stacked_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hints", action="store_true", help="Use hints.")
    args = parser.parse_args()

    # set seed
    pl.seed_everything(args.seed)
    logger.info(f"Using seed {args.seed}")

    # load config
    cfg = load_cfg(args.cfg)

    if args.hints:
        cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT = 1.0
        cfg.RUN_NAME = cfg.RUN_NAME+"-hints"
        logger.info("Using hints.")

    
    logger.info("Starting run...")
    torch.set_float32_matmul_precision('medium')

    # load datasets
    train_ds = CLRSDataset(algorithm="bfs", split="train", num_samples=1000)
    val_ds = CLRSDataset(algorithm="bfs", split="val", num_samples=32)
    test_ds = CLRSDataset(algorithm="bfs", split="test", num_samples=32) 
    specs = train_ds.specs
    
    # load model
    datamodule = CLRSDataModule(train_dataset=train_ds,val_datasets=val_ds, test_datasets=test_ds, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS, test_batch_size=cfg.TEST.BATCH_SIZE)
    datamodule.val_dataloader()
    print(train_ds.specs)
    model = SALSACLRSModel(specs=train_ds.specs, cfg=cfg)

    ckpt_dir = "./saved/"
    train(model, datamodule, cfg, train_ds.specs, seed = args.seed, checkpoint_dir=ckpt_dir)
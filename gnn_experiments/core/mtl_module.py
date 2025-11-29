import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import defaultdict
from loguru import logger
from sklearn.metrics import f1_score

from .models import EncodeProcessDecode, MultitaskEncodeProcessDecode, MMOE_EncodeProcessDecode
from .loss import CLRSLoss
from .utils import stack_dicts
from .metrics import calc_metrics

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

class MultiCLRSModel(pl.LightningModule):
    def __init__(self, task_to_specs, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.task_to_specs = task_to_specs
        self.model = model
        self.task_to_losses = {task_name: CLRSLoss(specs, cfg.TRAIN.LOSS.HIDDEN_LOSS_TYPE) for task_name, specs in task_to_specs.items()}
        self.step_output_cache = defaultdict(list)

    def forward(self, batch):
        return self.model(batch)        

    def _loss(self, batch, output, hints, hidden):
        task_name = batch["task_name"]; batch = batch["data"]
        outloss, hintloss, hiddenloss = self.task_to_losses[task_name](batch, output, hints, hidden)
        loss = self.cfg.TRAIN.LOSS.OUTPUT_LOSS_WEIGHT * outloss + self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT * hintloss + self.cfg.TRAIN.LOSS.HIDDEN_LOSS_WEIGHT * hiddenloss
        return loss, outloss, hintloss, hiddenloss

    def training_step(self, batch, batch_idx):
        output, hints, hidden = self.model(batch)
        loss, outloss, hintloss, hiddenloss = self._loss(batch, output, hints, hidden)
        self.log("train/outloss", outloss, batch_size=batch["data"].num_graphs, prog_bar=True)
        self.log("train/hintloss", hintloss, batch_size=batch["data"].num_graphs, prog_bar=True)
        self.log("train/hiddenloss", hiddenloss, batch_size=batch["data"].num_graphs)
        self.log("train/loss", loss, batch_size=batch["data"].num_graphs, prog_bar=True)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def _shared_eval(self, batch):
        output, hints, hidden = self.model(batch)
        loss, outloss, hintloss, hiddenloss = self._loss(batch, output, hints, hidden)
        # calc batch metrics
        task_name = batch["task_name"]; batch = batch["data"]
        # assert len(batch.outputs) == 1
        metrics = calc_metrics(batch.outputs[0], output, batch, self.task_to_specs[task_name][batch.outputs[0]][2], loc=self.task_to_specs[task_name][batch.outputs[0]][1])
        output.update({f"{m}_metric": metrics[m] for m in metrics})
        output["batch_size"] = torch.tensor(batch.num_graphs).float()
        output["num_nodes"] = torch.tensor(batch.num_nodes).float()
        output["loss"] = loss.cpu()
        return loss, output

    def _end_of_epoch_metrics(self, task_name):
        output = stack_dicts(self.step_output_cache[task_name])
        # average metrics over graphs
        metrics = {}
        for m in output:
            if not m.endswith("_metric"):
                continue
            if m.startswith("graph"):
                # graph level metrics have to be computed differently
                metrics["graph_accuracy"] = output[m].float().mean()
                metrics["graph_f1"] = f1_score(torch.ones_like(output[m]).cpu().numpy(), output[m].cpu().numpy(), average='binary')
            else:
                metrics[m[:-7]] = output[m].float().mean()
        return metrics
    
    def validation_step(self, batch, batch_idx,):
        task_name = batch["task_name"]
        loss, output = self._shared_eval(batch)
        self.log(f'val_loss', loss, batch_size=batch["data"].num_graphs)

        self.step_output_cache[task_name].append(output)
        return loss
    
    def test_step(self, batch, batch_idx,):
        task_name = batch["task_name"]
        loss, output = self._shared_eval(batch)
        self.log(f'test_loss', loss, batch_size=batch["data"].num_graphs)

        self.step_output_cache[task_name].append(output)
        return loss
    
    def on_validation_epoch_end(self):
        summary = defaultdict(list)
        for task_name in self.step_output_cache.keys():
            metrics = self._end_of_epoch_metrics(task_name)
            for key in metrics:
                self.log(f"val_{task_name}_{key}", metrics[key])
                summary[f"val_{task_name}_{key}"].append(metrics[key])
        for key in ["node_accuracy", "graph_accuracy", "graph_f1"]:
            summary[key] = np.concatenate([v for k, v in summary.items() if key in k])
            for val in summary[key]:
                self.log(f"val_{key}", val)

        for key in summary:
            summary[key] = np.mean(summary[key])
        print(summary)
        self.step_output_cache.clear()

    def on_test_epoch_end(self):
        summary = defaultdict(list)

        for task_name in self.step_output_cache.keys():
            metrics = self._end_of_epoch_metrics(task_name)
            for key in metrics:
                self.log(f"test_{task_name}_{key}", metrics[key])
                summary[f"test_{task_name}_{key}"].append(metrics[key])
        for key in ["node_accuracy", "graph_accuracy", "graph_f1"]:
            summary[key] = np.concatenate([v for k, v in summary.items() if key in k])
            for val in summary[key]:
                self.log(f"test_{key}", val)

        for key in summary:
            summary[key] = np.mean(summary[key])
        print(summary)
        self.step_output_cache.clear()  

    def configure_optimizers(self):
        if self.cfg.TRAIN.OPTIMIZER.NAME == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.TRAIN.OPTIMIZER.LR)
        elif self.cfg.TRAIN.OPTIMIZER.NAME == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.OPTIMIZER.LR)
        else:
            raise NotImplementedError(f"Optimizer {self.cfg.TRAIN.OPTIMIZER.NAME} not implemented")
        out = {"optimizer": optimizer, "monitor": "val_loss", "interval": "step", "frequency": 1}
        if self.cfg.TRAIN.SCHEDULER.ENABLE:
            try:
                scheduler = getattr(torch.optim.lr_scheduler, self.cfg.TRAIN.SCHEDULER.NAME)(optimizer, **self.cfg.TRAIN.SCHEDULER.PARAMS[0])
                out["lr_scheduler"] = scheduler
                out['monitor'] = 'val_loss'
                
            except AttributeError:
                raise NotImplementedError(f"Scheduler {self.cfg.TRAIN.SCHEDULER.NAME} not implemented")

        return out


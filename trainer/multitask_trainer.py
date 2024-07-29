import os
import numpy as np
import numpy
import torch
from numpy import inf
from utils import MetricTracker, prepare_inputs
import torch.nn.functional as F
from compute_metrics import exact_match_score, accuracy_score, metric_max_over_ground_truths, edit_distance_score

def compute_accuracy(predictions, references, task_indices=None):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    xlingual = False
    if task_indices is None:
        task_indices = [None] * len(predictions)
        metrics = {"accuracy": 0, "edit_distance": 0}
    else:
        metrics = {f"{task_name}_accuracy": 0 for task_name in task_indices[0].keys()}
        metrics.update({f"{task_name}_edit_distance": 0 for task_name in task_indices[0].keys()})
        metrics.update({f"{task_name}_num_samples": 0 for task_name in task_indices[0].keys()})
    
    for pred, gold, task_idx in zip(predictions, references, task_indices):
        assert isinstance(gold, list)
        if task_idx is None:
            metrics["accuracy"] += metric_max_over_ground_truths(
                exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
            metrics["edit_distance"] += metric_max_over_ground_truths(
                edit_distance_score, prediction=pred, ground_truths=gold, xlingual=xlingual
            )
        else:
            # task_idx is dict
            for task_name, idx in task_idx.items():
                if len(idx) == 0:
                    continue
                tmp_accuracy = metric_max_over_ground_truths(
                    exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual, indices = idx
                )
                tmp_edit_distance = metric_max_over_ground_truths(
                    edit_distance_score, prediction=pred, ground_truths=gold, xlingual=xlingual, indices = idx
                )
                metrics[f"{task_name}_accuracy"] += tmp_accuracy
                metrics[f"{task_name}_edit_distance"] += tmp_edit_distance
                metrics[f"{task_name}_num_samples"] += 1
    
    if task_indices is None:
        metrics["accuracy"] = 100.0 * metrics["accuracy"] / len(references)
        metrics["edit_distance"] = metrics["edit_distance"] / len(references)
    else:
        for key, val in metrics.items():
            if "accuracy" in key:
                task_name = "_".join(key.split("_")[:-1])
                metrics[key] = (100.0 * val / metrics[f"{task_name}_num_samples"]) if metrics[f"{task_name}_num_samples"] > 0 else 0
            elif "edit_distance" in key:
                task_name = "_".join(key.split("_")[:-2])
                metrics[key] = (val / metrics[f"{task_name}_num_samples"]) if metrics[f"{task_name}_num_samples"] > 0 else 0
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

class MultitaskTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, tokenizer, optimizer, lr_scheduler,
                 config, device, logger,
                 multitask_train_data_loader,
                 train_data_loaders,
                 valid_data_loaders,
                 test_data_loaders=None, 
                 checkpoint_dir=None,
                 criterion=None,
                 generate_length=5,
                 save_intial_model=True,
                 eval_epoch = 1,
                 task_output_columns = None,
                 model_use_task_name = False, # whether to pass task name to the model, True if we use multihead model
                 ):
        self.config = config
        self.logger = logger
        self.device = device

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.criterion = criterion
        # self.metric = metric
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.cfg_trainer = config['trainer']
        self.epochs = self.cfg_trainer['num_train_epochs']
        self.save_period = self.cfg_trainer['save_period']
        self.monitor = self.cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.completed_steps = 0
        self.eval_epoch = eval_epoch
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is None:
            self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.multitask_train_data_loader = multitask_train_data_loader
        self.train_data_loaders = train_data_loaders
        self.valid_data_loaders = valid_data_loaders
        self.test_data_loaders = test_data_loaders if test_data_loaders is not None else valid_data_loaders
        self.task_output_columns = task_output_columns
        self.do_validation = self.valid_data_loaders is not None
        self.len_epoch = len(self.multitask_train_data_loader)

        self.train_metrics = MetricTracker('loss')
        self.valid_metrics = {}
        for task_name in self.train_data_loaders.keys():
            if task_output_columns is not None:
                for column in task_output_columns[task_name]:
                    column_name = f"column_{column}"
                    self.valid_metrics[f"{task_name}_{column_name}"] = MetricTracker('loss', 'accuracy', 'edit_distance')
            else:
                self.valid_metrics[task_name] = MetricTracker('loss', 'accuracy', 'edit_distance')
        self.valid_metrics['average'] = MetricTracker('loss', 'accuracy', 'edit_distance')
        self.generate_length = generate_length
        
        if save_intial_model:
            self._save_checkpoint(epoch=0, name="model_epoch_0")

        self.model_use_task_name = model_use_task_name

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for step, batch in enumerate(self.multitask_train_data_loader):
            task_name = batch['task_name']
            batch = batch['data']
            inputs_batch = {k: v for k, v in batch.items() if (k not in ["inputs", "steps", "indexes", "task_indices"])}
            inputs_batch = prepare_inputs(inputs_batch, self.device)

            outputs = self.model(**inputs_batch, task_name=task_name) if self.model_use_task_name else self.model(**inputs_batch)
            loss = outputs.loss
            loss = loss / self.cfg_trainer["gradient_accumulation_steps"]
            loss.backward()
            if step % self.cfg_trainer["gradient_accumulation_steps"] == 0 or step == len(self.train_data_loader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            self.completed_steps += 1
            if self.completed_steps >= self.cfg_trainer["max_train_steps"]:
                break

            # update training metrics
            self.train_metrics.update('loss', loss.item())

        log = self.train_metrics.result()

        if self.do_validation and epoch % self.eval_epoch == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        log = {}
        self.valid_metrics['average'].reset()
        for task_name in self.valid_metrics.keys():
            self.valid_metrics[task_name].reset()
        for task_name, valid_data_loader in self.valid_data_loaders.items():
            output_columns = self.task_output_columns[task_name] if self.task_output_columns is not None else None
            for step, batch in enumerate(valid_data_loader):
                inputs_batch = {k: v for k, v in batch.items() if (k not in ["inputs", "steps", "indexes", "task_indices"])}
                inputs_batch = prepare_inputs(inputs_batch, self.device)
                outputs = self.model(**inputs_batch, task_name=task_name) if self.model_use_task_name else self.model(**inputs_batch)
                if step == 0:
                    print("inputs", self.tokenizer.batch_decode(batch["input_ids"][:8], skip_special_tokens=True))

                """
                generate response
                """
                # Get the labels
                gold_answers = batch["labels"].clone()
                gold_answers[gold_answers == -100] = self.tokenizer.pad_token_id
                output_len = (gold_answers != self.tokenizer.pad_token_id).sum(dim=1).max().item()
                gold_answers = self.tokenizer.batch_decode(gold_answers, skip_special_tokens=True)

                # get only the inputs
                inputs = batch["input_ids"].clone()
                inputs[batch["labels"] != -100] = self.tokenizer.pad_token_id
                inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)
                self.tokenizer.padding_side = 'left'
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                self.tokenizer.padding_side = 'right'
                inputs = prepare_inputs(inputs, self.device)
                if "position_ids" in batch:
                    inputs["position_ids"] = torch.zeros_like(inputs["input_ids"])
                
                if self.model_use_task_name:
                    inputs["task_name"] = task_name
                generated = self.model.generate(**inputs, 
                                                max_new_tokens=self.generate_length, 
                                                pad_token_id=self.tokenizer.pad_token_id)

                # remove the given prompt in the output
                input_len = inputs["input_ids"].shape[1]
                generated[:, :input_len] = self.tokenizer.pad_token_id
                generated[:, input_len+output_len:] = self.tokenizer.pad_token_id
                pred_answers = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                gold_answers = [[answer] for answer in gold_answers]
                if step == 0:
                    print("gold_answers", gold_answers[:8])
                    print("pred_answers", pred_answers[:8])
                metrics = compute_accuracy(pred_answers, gold_answers, task_indices=batch["task_indices"])

                if output_columns is None:
                    self.valid_metrics[task_name].update('loss', outputs.loss.item(), n=len(batch["labels"]))
                    self.valid_metrics[task_name].update('accuracy', metrics["accuracy"], n=len(batch["labels"]))
                    self.valid_metrics[task_name].update('edit_distance', metrics["edit_distance"], n=len(batch["labels"]))

                    self.valid_metrics['average'].update('loss', outputs.loss.item(), n=len(batch["labels"]))
                    self.valid_metrics['average'].update('accuracy', metrics["accuracy"], n=len(batch["labels"]))
                    self.valid_metrics['average'].update('edit_distance', metrics["edit_distance"], n=len(batch["labels"]))
                    continue

                for column in output_columns:
                    column_name = f"column_{column}"
                    self.valid_metrics[f"{task_name}_{column_name}"].update('loss', outputs.loss.item(), n=len(batch["labels"]))
                    self.valid_metrics[f"{task_name}_{column_name}"].update('accuracy', metrics[f"{column_name}_accuracy"], n=metrics[f"{column_name}_num_samples"])
                    self.valid_metrics[f"{task_name}_{column_name}"].update('edit_distance', metrics[f"{column_name}_edit_distance"], n=metrics[f"{column_name}_num_samples"])

                    self.valid_metrics['average'].update('loss', outputs.loss.item(), n=len(batch["labels"]))
                    self.valid_metrics['average'].update('accuracy', metrics[f"{column_name}_accuracy"], n=metrics[f"{column_name}_num_samples"])
                    self.valid_metrics['average'].update('edit_distance', metrics[f"{column_name}_edit_distance"], n=metrics[f"{column_name}_num_samples"])
            
            if output_columns is None:
                task_log = self.valid_metrics[task_name].result()
                log.update({task_name + '_' + k : v for k, v in task_log.items()})
            else:
                for column in output_columns:
                    column_name = f"column_{column}"
                    task_log = self.valid_metrics[f"{task_name}_{column_name}"].result()
                    log.update({f"{task_name}_{column_name}_" + k : v for k, v in task_log.items()})
        avg_log = self.valid_metrics['average'].result()
        log.update({k : v for k, v in avg_log.items()})
        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        log = {}
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    # self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if improved:
                self._save_checkpoint(epoch)
        return log

    @torch.no_grad()
    def test(self, load_best = True):
        if self.test_data_loaders is None:
            self.logger.info("No test data set.")
            return

        if load_best:
            best_path = os.path.join(self.checkpoint_dir, f'model_best.pth')
            if os.path.exists(best_path):
                self.model.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])
                print("Load best checkpoint from {}".format(best_path))

        self.model.eval()

        log = {}
        for task_name in self.valid_metrics.keys():
            self.valid_metrics[task_name].reset()
        for task_name, test_data_loader in self.test_data_loaders.items():
            output_columns = self.task_output_columns[task_name] if self.task_output_columns is not None else None
            for step, batch in enumerate(test_data_loader):
                inputs_batch = {k: v for k, v in batch.items() if (k not in ["inputs", "steps", "indexes", "task_indices"])}
                inputs_batch = prepare_inputs(inputs_batch, self.device)
                outputs = self.model(**inputs_batch, task_name=task_name) if self.model_use_task_name else self.model(**inputs_batch)
                if step == 0:
                    print("inputs", self.tokenizer.batch_decode(batch["input_ids"][:8], skip_special_tokens=True))

                """
                generate response
                """
                # Get the labels
                gold_answers = batch["labels"].clone()
                gold_answers[gold_answers == -100] = self.tokenizer.pad_token_id
                output_len = (gold_answers != self.tokenizer.pad_token_id).sum(dim=1).max().item()
                gold_answers = self.tokenizer.batch_decode(gold_answers, skip_special_tokens=True)

                # get only the inputs
                inputs = batch["input_ids"].clone()
                inputs[batch["labels"] != -100] = self.tokenizer.pad_token_id
                inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)
                self.tokenizer.padding_side = 'left'
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                self.tokenizer.padding_side = 'right'
                inputs = prepare_inputs(inputs, self.device)
                if "position_ids" in batch:
                    inputs["position_ids"] = torch.zeros_like(inputs["input_ids"])
                
                if self.model_use_task_name:
                    inputs["task_name"] = task_name
                generated = self.model.generate(**inputs, 
                                                max_new_tokens=self.generate_length, 
                                                pad_token_id=self.tokenizer.pad_token_id)

                # remove the given prompt in the output
                input_len = inputs["input_ids"].shape[1]
                generated[:, :input_len] = self.tokenizer.pad_token_id
                generated[:, input_len+output_len:] = self.tokenizer.pad_token_id
                pred_answers = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                gold_answers = [[answer] for answer in gold_answers]
                if step == 0:
                    print("gold_answers", gold_answers[:8])
                    print("pred_answers", pred_answers[:8])
                metrics = compute_accuracy(pred_answers, gold_answers, task_indices=batch["task_indices"])

                if output_columns is None:
                    self.valid_metrics[task_name].update('loss', outputs.loss.item(), n=len(batch["labels"]))
                    self.valid_metrics[task_name].update('accuracy', metrics["accuracy"], n=len(batch["labels"]))
                    self.valid_metrics[task_name].update('edit_distance', metrics["edit_distance"], n=len(batch["labels"]))

                    self.valid_metrics['average'].update('loss', outputs.loss.item(), n=len(batch["labels"]))
                    self.valid_metrics['average'].update('accuracy', metrics["accuracy"], n=len(batch["labels"]))
                    self.valid_metrics['average'].update('edit_distance', metrics["edit_distance"], n=len(batch["labels"]))
                    continue

                for column in output_columns:
                    column_name = f"column_{column}"
                    self.valid_metrics[f"{task_name}_{column_name}"].update('loss', outputs.loss.item(), n=len(batch["labels"]))
                    self.valid_metrics[f"{task_name}_{column_name}"].update('accuracy', metrics[f"{column_name}_accuracy"], n=metrics[f"{column_name}_num_samples"])
                    self.valid_metrics[f"{task_name}_{column_name}"].update('edit_distance', metrics[f"{column_name}_edit_distance"], n=metrics[f"{column_name}_num_samples"])

                    self.valid_metrics['average'].update('loss', outputs.loss.item(), n=len(batch["labels"]))
                    self.valid_metrics['average'].update('accuracy', metrics[f"{column_name}_accuracy"], n=metrics[f"{column_name}_num_samples"])
                    self.valid_metrics['average'].update('edit_distance', metrics[f"{column_name}_edit_distance"], n=metrics[f"{column_name}_num_samples"])
            
            if output_columns is None:
                task_log = self.valid_metrics[task_name].result()
                log.update({task_name + '_' + k : v for k, v in task_log.items()})
            else:
                for column in output_columns:
                    column_name = f"column_{column}"
                    task_log = self.valid_metrics[f"{task_name}_{column_name}"].result()
                    log.update({f"{task_name}_{column_name}_" + k : v for k, v in task_log.items()})
        avg_log = self.valid_metrics['average'].result()
        log.update({k : v for k, v in avg_log.items()})
        log = {f"test_{k}": v for k, v in log.items()}
        self.logger.info(log)
        return log

    
    def load_best_checkpoint(self):
        best_path = os.path.join(self.checkpoint_dir, f'model_best.pth')
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"], strict=False)
            print("Load best checkpoint from {}".format(best_path))

    def _save_checkpoint(self, epoch, name = "model_best"):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
        }
        
        best_path = os.path.join(self.checkpoint_dir, f'{name}.pth')
        torch.save(state, best_path)
        self.logger.info(f"Saving current model: {name}.pth ...")

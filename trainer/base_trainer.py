import os
import numpy as np
import numpy
import torch
from numpy import inf
from utils import MetricTracker, prepare_inputs
import torch.nn.functional as F
from compute_metrics import exact_match_score, accuracy_score, metric_max_over_ground_truths, edit_distance_score

def compute_accuracy(predictions, references, indices=None):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    xlingual = False
    accuracy = 0; edit_distance = 0
    for i, (pred, gold) in enumerate(zip(predictions, references)):
        assert isinstance(gold, list)
        accuracy += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual, indices=indices[i] if indices is not None else None
        )
        edit_distance += metric_max_over_ground_truths(
            edit_distance_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    accuracy = 100.0 * accuracy / len(references)
    edit_distance = edit_distance / len(references)
    metrics = {"accuracy": accuracy, "edit_distance": edit_distance}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

class Trainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, tokenizer, optimizer, lr_scheduler,
                 config, device, logger,
                 train_data_loader, 
                 valid_data_loader=None,
                 test_data_loader=None, 
                 checkpoint_dir=None,
                 criterion=None,
                 generate_length=5,
                 save_intial_model=True, 
                 compute_metrics_for_steps=False,
                 max_inter_steps=10):
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
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is None:
            self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.len_epoch = len(self.train_data_loader)

        self.train_metrics = MetricTracker('loss')
        self.valid_metrics = MetricTracker('loss', 'accuracy', 'edit_distance')
        self.step_valid_metrics = None
        if compute_metrics_for_steps:
            self.max_inter_steps = max_inter_steps
            self.step_valid_metrics = []
            for _ in range(max_inter_steps):
                self.step_valid_metrics.append(MetricTracker('accuracy', 'edit_distance'))
        self.generate_length = generate_length
        
        if save_intial_model:
            self._save_checkpoint(epoch=0, name="model_epoch_0")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for step, batch in enumerate(self.train_data_loader):
            batch = prepare_inputs(batch, self.device)
            inputs_batch = {k: v for k, v in batch.items() if (k not in ["inputs", "steps", "indexes"])}
            outputs = self.model(**inputs_batch)
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

        if self.do_validation:
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
        self.valid_metrics.reset()
        print(self.step_valid_metrics is not None)
        if self.step_valid_metrics is not None:
            for i in range(self.max_inter_steps):
                self.step_valid_metrics[i].reset()

        for step, batch in enumerate(self.valid_data_loader):
            batch = prepare_inputs(batch, self.device)
            inputs_batch = {k: v for k, v in batch.items() if (k not in ["inputs", "steps", "indexes"])}
            outputs = self.model(**inputs_batch)
            if step == 0:
                labels = outputs.logits.detach()
                labels = torch.argmax(labels, dim=-1)

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
            generated = self.model.generate(**inputs, max_new_tokens=self.generate_length, pad_token_id=self.tokenizer.pad_token_id)

            # remove the given prompt in the output
            input_len = inputs["input_ids"].shape[1]
            generated[:, :input_len] = self.tokenizer.pad_token_id
            generated[:, input_len+output_len:] = self.tokenizer.pad_token_id
            pred_answers = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            gold_answers = [[answer] for answer in gold_answers]
            if step == 0:
                print("gold_answers", gold_answers[:8])
                print("pred_answers", pred_answers[:8])
            metrics = compute_accuracy(pred_answers, gold_answers, indices=batch.get("indexes", None))

            self.valid_metrics.update('loss', outputs.loss.item())
            self.valid_metrics.update('accuracy', metrics["accuracy"])
            self.valid_metrics.update('edit_distance', metrics["edit_distance"])
            if self.step_valid_metrics is not None:
                steps = batch["steps"]; assert steps is not None
                for i in range(self.max_inter_steps):
                    step_preds = [pred_answers[j] for j, step in enumerate(steps) if step == (i+1)]
                    step_answers = [gold_answers[j] for j, step in enumerate(steps) if step == (i+1)]
                    step_indexes = None
                    if "indexes" in batch:
                        step_indexes = [batch["indexes"][j] for j, step in enumerate(steps) if step == (i+1)]
                    if len(step_preds) == 0: continue
                    metrics = compute_accuracy(step_preds, step_answers, indices=step_indexes)
                    # print(f"step {i+1} preds", len(step_preds), metrics)
                    self.step_valid_metrics[i].update('accuracy', metrics["accuracy"], n=len(step_preds))
                    self.step_valid_metrics[i].update('edit_distance', metrics["edit_distance"], n=len(step_preds))
        
        log = self.valid_metrics.result()
        if self.step_valid_metrics is not None:
            for i in range(self.max_inter_steps):
                step_log = self.step_valid_metrics[i].result()
                log.update(**{f'step_{i+1}_'+k : v for k, v in step_log.items()})
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
                    self.mnt_mode = 'off'
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
        if self.test_data_loader is None:
            self.logger.info("No test data set.")
            return

        if load_best:
            best_path = os.path.join(self.checkpoint_dir, f'model_best.pth')
            if os.path.exists(best_path):
                self.model.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])
                print("Load best checkpoint from {}".format(best_path))

        self.model.eval()
        self.valid_metrics.reset()
        if self.step_valid_metrics is not None:
            for i in range(self.max_inter_steps):
                self.step_valid_metrics[i].reset()
        for step, batch in enumerate(self.test_data_loader):
            batch = prepare_inputs(batch, self.device)
            inputs_batch = {k: v for k, v in batch.items() if (k not in ["inputs", "steps", "indexes"])}
            outputs = self.model(**inputs_batch)

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
            generated = self.model.generate(**inputs, max_new_tokens=self.generate_length, pad_token_id=self.tokenizer.pad_token_id)

            # remove the given prompt in the output
            input_len = inputs["input_ids"].shape[1]
            generated[:, :input_len] = self.tokenizer.pad_token_id
            generated[:, input_len+output_len:] = self.tokenizer.pad_token_id
            pred_answers = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            gold_answers = [[answer] for answer in gold_answers]
            if step == 0:
                print("gold_answers", gold_answers[:8])
                print("pred_answers", pred_answers[:8])
            metrics = compute_accuracy(pred_answers, gold_answers, indices=batch.get("indexes", None))
            self.valid_metrics.update('loss', outputs.loss.item())
            self.valid_metrics.update('accuracy', metrics["accuracy"])
            self.valid_metrics.update('edit_distance', metrics["edit_distance"])
            if self.step_valid_metrics is not None:
                steps = batch["steps"]; assert steps is not None
                for i in range(self.max_inter_steps):
                    step_preds = [pred_answers[j] for j, step in enumerate(steps) if step == (i+1)]
                    step_answers = [gold_answers[j] for j, step in enumerate(steps) if step == (i+1)]
                    step_indexes = None
                    if "indexes" in batch:
                        step_indexes = [batch["indexes"][j] for j, step in enumerate(steps) if step == (i+1)]
                    if len(step_preds) == 0: continue
                    metrics = compute_accuracy(step_preds, step_answers, indices=step_indexes)
                    self.step_valid_metrics[i].update('accuracy', metrics["accuracy"], n=len(step_preds))
                    self.step_valid_metrics[i].update('edit_distance', metrics["edit_distance"], n=len(step_preds))
        
        log = self.valid_metrics.result()
        if self.step_valid_metrics is not None:
            for i in range(self.max_inter_steps):
                step_log = self.step_valid_metrics[i].result()
                log.update(**{f'step_{i+1}_'+k : v for k, v in step_log.items()})
        self.logger.info(log)
        return log

    # old version
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
        # self.logger.info("Best checkpoint in epoch {}".format(epoch))
        
        best_path = os.path.join(self.checkpoint_dir, f'{name}.pth')
        torch.save(state, best_path)
        self.logger.info(f"Saving current model: {name}.pth ...")

    # old version
    def load_best_model(self):
        # Load the best model then test
        arch = type(self.model).__name__
        best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        state_dict  = torch.load(best_path, map_location=self.device)["state_dict"]
        self.model.load_state_dict(state_dict)
        return state_dict

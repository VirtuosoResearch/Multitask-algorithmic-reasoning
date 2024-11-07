from trainer import Trainer, MultitaskTrainer
import torch
import torch.nn.functional as F
from utils import prepare_inputs

class LabelSmoothTrainer(Trainer):

    def __init__(self, model, tokenizer, optimizer, lr_scheduler, config, device, logger, 
                 train_data_loader, valid_data_loader=None, test_data_loader=None, checkpoint_dir=None, 
                 criterion=None, generate_length=5, save_intial_model=True,
                 alpha = 0, num_classes = 2, **kwargs):
        super().__init__(model, tokenizer, optimizer, lr_scheduler, config, device, logger, 
                         train_data_loader, valid_data_loader, test_data_loader, checkpoint_dir, 
                         criterion, generate_length, save_intial_model, **kwargs)
        self.alpha = alpha
        self.num_classes = num_classes
        self.smoothed_label = torch.ones(1, num_classes, dtype=torch.float).to(device) 
        self.smoothed_label = self.smoothed_label / num_classes

    def _label_smooth_loss(self, output, target):
        shift_logits = output[:, :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = target[:, 1:].contiguous().view(-1)
        ce_loss = F.nll_loss(shift_logits, shift_labels, ignore_index=-100)
        smooth_loss = torch.sum(- shift_logits[shift_labels != -100, :] * self.smoothed_label, dim=1).mean()
        loss = ce_loss * (1-self.alpha) + smooth_loss * self.alpha
        return loss
    
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
            log_probs = F.log_softmax(outputs.logits, dim=2)
            labels = batch["labels"]
            loss = self._label_smooth_loss(log_probs, labels)
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
    
class LabelSmoothMultitaskTrainer(MultitaskTrainer):

    def __init__(self, model, tokenizer, optimizer, lr_scheduler, config, device, logger, 
                 multitask_train_data_loader, train_data_loaders, valid_data_loaders, test_data_loaders=None, 
                 checkpoint_dir=None, criterion=None, generate_length=5, save_intial_model=True, eval_epoch=1,
                 alpha = 0, num_classes = 2,):
        super().__init__(model, tokenizer, optimizer, lr_scheduler, config, device, logger, 
                         multitask_train_data_loader, train_data_loaders, valid_data_loaders, test_data_loaders, 
                         checkpoint_dir, criterion, generate_length, save_intial_model, eval_epoch)
        
        self.alpha = alpha
        self.num_classes = num_classes
        self.smoothed_label = torch.ones(1, num_classes, dtype=torch.float).to(device) 
        self.smoothed_label = self.smoothed_label / num_classes

    def _label_smooth_loss(self, output, target):
        shift_logits = output[:, :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = target[:, 1:].contiguous().view(-1)
        ce_loss = F.nll_loss(shift_logits, shift_labels, ignore_index=-100)
        smooth_loss = torch.sum(- shift_logits[shift_labels != -100, :] * self.smoothed_label, dim=1).mean()
        loss = ce_loss * (1-self.alpha) + smooth_loss * self.alpha
        return loss
    
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

            batch = prepare_inputs(batch, self.device)
            inputs_batch = {k: v for k, v in batch.items() if (k not in ["inputs", "steps", "indexes"])}
            outputs = self.model(**inputs_batch, task_name=task_name)
            # loss = outputs.loss
            log_probs = F.log_softmax(outputs.logits, dim=2)
            labels = batch["labels"]
            loss = self._label_smooth_loss(log_probs, labels)
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
from trainer import Trainer, MultitaskTrainer
from utils import prepare_inputs
from utils.new_constraint_v2 import TopKFrobeniusConstraint, UniformFrobeniusConstraint
from utils.new_constraint import OldTopKFrobeniusConstraint

class ConstraintAllocationTrainer(Trainer):

    def __init__(self, model, tokenizer, optimizer, lr_scheduler, 
                 config, device, logger, train_data_loader, 
                 valid_data_loader=None, test_data_loader=None, checkpoint_dir=None, criterion=None, generate_length=5, **kwargs):
        super().__init__(model, tokenizer, optimizer, lr_scheduler, 
                config, device, logger, train_data_loader, 
                valid_data_loader, test_data_loader, checkpoint_dir, criterion, generate_length, **kwargs)
    
        self.constraints = []
        self.penalty = []

    def add_constraint(self, constraint_lambda, state_dict = None, allocation_method = "topk"):
        type_model = type(self.model)
        if allocation_method == "topk":
            self.constraints.append(
                TopKFrobeniusConstraint(type_model, constraint_lambda, 
                    state_dict = state_dict, including_key=["c_attn.weight_q", "c_attn.weight_k", "c_attn.weight_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]),
            )
        elif allocation_method == "uniform":
            self.constraints.append(
                UniformFrobeniusConstraint(type_model, constraint_lambda, 
                    state_dict = state_dict, including_key=["c_attn.weight_q", "c_attn.weight_k", "c_attn.weight_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"])
            )

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

                """Apply Constraints"""
                # 1. compute the gradient 
                outputs = self.model(**inputs_batch)
                loss = outputs.loss
                loss.backward()

                # 2. allocate constraint according to weight-gradient product 
                for constraint in self.constraints:
                    self.model.apply(constraint)
                self.optimizer.zero_grad()
                """Apply Constraints"""

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
    
class ConstraintAllocationMultitaskTrainer(MultitaskTrainer):

    def __init__(self, model, tokenizer, optimizer, lr_scheduler, config, device, logger, 
                 multitask_train_data_loader, train_data_loaders, valid_data_loaders, test_data_loaders=None, 
                 checkpoint_dir=None, criterion=None, generate_length=5, save_intial_model=True, eval_epoch=1):
        super().__init__(model, tokenizer, optimizer, lr_scheduler, config, device, logger, 
                multitask_train_data_loader, train_data_loaders, valid_data_loaders, test_data_loaders, 
                checkpoint_dir, criterion, generate_length, save_intial_model, eval_epoch)
        
        self.constraints = []

    def add_constraint(self, constraint_lambda, state_dict = None, allocation_method = "topk",
                       alpha=1e-3, use_topk=False):
        type_model = type(self.model)
        if allocation_method == "topk":
            self.constraints.append(
                TopKFrobeniusConstraint(type_model, constraint_lambda, 
                    state_dict = state_dict, including_key=["c_attn.weight_q", "c_attn.weight_k", "c_attn.weight_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
                    alpha=alpha, use_topk=use_topk
                )
            )
        if allocation_method == "old_topk":
            self.constraints.append(
                OldTopKFrobeniusConstraint(type_model, constraint_lambda, 
                    state_dict = state_dict, including_key=["c_attn.weight_q", "c_attn.weight_k", "c_attn.weight_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]),
            )
        elif allocation_method == "uniform":
            self.constraints.append(
                UniformFrobeniusConstraint(type_model, constraint_lambda, 
                    state_dict = state_dict, including_key=["c_attn.weight_q", "c_attn.weight_k", "c_attn.weight_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"])
            )

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
            loss = outputs.loss
            loss = loss / self.cfg_trainer["gradient_accumulation_steps"]
            loss.backward()
            if step % self.cfg_trainer["gradient_accumulation_steps"] == 0 or step == len(self.train_data_loader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                """Apply Constraints"""
                # 1. compute the gradient 
                outputs = self.model(**inputs_batch, task_name=task_name)
                loss = outputs.loss
                loss.backward()

                # 2. allocate constraint according to weight-gradient product 
                for constraint in self.constraints:
                    self.model.apply(constraint)
                self.optimizer.zero_grad()
                """Apply Constraints"""
                
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
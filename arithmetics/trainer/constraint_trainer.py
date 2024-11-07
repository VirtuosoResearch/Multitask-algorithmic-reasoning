from trainer import Trainer, MultitaskTrainer
from utils import prepare_inputs
from utils.constraint import FrobeniusConstraint, add_penalty

class ConstraintTrainer(Trainer):

    def __init__(self, model, tokenizer, optimizer, lr_scheduler, 
                 config, device, logger, train_data_loader, 
                 valid_data_loader=None, test_data_loader=None, checkpoint_dir=None, criterion=None, generate_length=5, **kwargs):
        super().__init__(model, tokenizer, optimizer, lr_scheduler, 
                config, device, logger, train_data_loader, 
                valid_data_loader, test_data_loader, checkpoint_dir, criterion, generate_length, **kwargs)
    
        self.constraints = []
        self.penalty = []

    def add_constraint(self, lambda_q, lambda_k, lambda_v, lambda_linear_1,
                       lambda_linear_2, lambda_linear_3, state_dict = None):
        type_model = type(self.model)
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_q, 
                state_dict = state_dict, including_key="c_attn.weight_q")
        )
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_k, 
                state_dict = state_dict, including_key="c_attn.weight_k")
        )
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_v, 
                state_dict = state_dict, including_key="c_attn.weight_v")
        )
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_linear_1, 
                state_dict = state_dict, including_key="attn.c_proj")
        )
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_linear_2, 
                state_dict = state_dict, including_key="mlp.c_fc")
        )
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_linear_3, 
                state_dict = state_dict, including_key="mlp.c_proj")
        )
        # self.constraints.append(
        #     FrobeniusConstraint(type_model, lambda_pred_head, 
        #         state_dict = state_dict, including_key = "lm_head")
        # )

    def add_penalties(self, lambda_attention, lambda_linear, lambda_pred_head, state_dict=None):
        norm = "frob"
        self.penalty.append(
            {"norm": norm, 
            "_lambda": lambda_attention,
            "excluding_key": None,
            "including_key": "c_attn",
            "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
            "_lambda": lambda_linear,
            "excluding_key": None,
            "including_key": "c_fc",
            "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
            "_lambda": lambda_linear,
            "excluding_key": None,
            "including_key": "c_proj",
            "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
            "_lambda": lambda_pred_head,
            "excluding_key": None,
            "including_key": "lm_head",
            "state_dict": state_dict}
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

            """Apply Penalties"""
            for penal in self.penalty:
                loss += add_penalty(
                    self.model, 
                    penal["norm"], 
                    penal["_lambda"], 
                    excluding_key = penal["excluding_key"],
                    including_key = penal["including_key"],
                    state_dict=penal["state_dict"]
                )
            """Apply Penalties"""

            loss.backward()
            if step % self.cfg_trainer["gradient_accumulation_steps"] == 0 or step == len(self.train_data_loader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                """Apply Constraints"""
                for constraint in self.constraints:
                    self.model.apply(constraint)
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
    

class ConstraintMultitaskTrainer(MultitaskTrainer):

    def __init__(self, model, tokenizer, optimizer, lr_scheduler, config, device, logger, 
                 multitask_train_data_loader, train_data_loaders, valid_data_loaders, test_data_loaders=None, 
                 checkpoint_dir=None, criterion=None, generate_length=5, save_intial_model=True, eval_epoch=1):
        super().__init__(model, tokenizer, optimizer, lr_scheduler, config, device, logger, 
                         multitask_train_data_loader, train_data_loaders, valid_data_loaders, test_data_loaders, 
                         checkpoint_dir, criterion, generate_length, save_intial_model, eval_epoch)
        
        self.constraints = []
    
    def add_constraint(self, lambda_q, lambda_k, lambda_v, lambda_linear_1,
                       lambda_linear_2, lambda_linear_3, state_dict = None):
        type_model = type(self.model)
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_q, 
                state_dict = state_dict, including_key="c_attn.weight_q")
        )
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_k, 
                state_dict = state_dict, including_key="c_attn.weight_k")
        )
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_v, 
                state_dict = state_dict, including_key="c_attn.weight_v")
        )
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_linear_1, 
                state_dict = state_dict, including_key="attn.c_proj")
        )
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_linear_2, 
                state_dict = state_dict, including_key="mlp.c_fc")
        )
        self.constraints.append(
            FrobeniusConstraint(type_model, lambda_linear_3, 
                state_dict = state_dict, including_key="mlp.c_proj")
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
                for constraint in self.constraints:
                    self.model.apply(constraint)
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
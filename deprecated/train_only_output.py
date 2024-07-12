import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_scheduler
from data_loader import generate_algorithm_prompt, format_algorithm_example, \
    generate_simple_algorithm_example, CLMCollator
from datasets import Dataset
from torch.utils.data import DataLoader
import datasets
import transformers

from trainer import *
import argparse
from parse_config import ConfigParser
import collections
import math
import os
import copy
from collections import defaultdict
import numpy as np

from utils.sam import SAM
from utils.adjustment import split_gpt_self_attention
from utils import interpolate_models, deep_copy
from utils.mixout import replace_layer_for_mixout, recursive_setattr

def load_model(model_name, if_pretrained=True, num_layers=12, tokenizer_dir="gpt2_sort_100"):
    """
    Return the model and tokenizer
    """
    if model_name in ['gpt2', "gpt2-medium", "gpt2-large", "gpt2-xl",]:
        if if_pretrained:
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                        # torch_dtype=torch.float16, 
                        #  ignore_mismatched_sizes=True
                        )
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False) # padding_side="left"
        else:
            tokenizer = AutoTokenizer.from_pretrained(f"./tokenizers/{tokenizer_dir}", use_fast=True) #  padding_side="left"
            config = AutoConfig.from_pretrained("gpt2")
            config.n_layer = num_layers
            config.vocab_size = tokenizer.vocab_size
            print("Num layers: {} Vocab size: {}".format(config.n_layer, config.vocab_size))
            model = AutoModelForCausalLM.from_config(config)
            
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

    return model, tokenizer

def load_algorithmic_dataset(algorithm, data_dir, train_size, valid_size, test_size, num_of_instances=1000000):
    file_name = f"./data/{algorithm}/{data_dir}.csv"
    instance_df = pd.read_csv(file_name, index_col=0)
    num_of_instances = min(num_of_instances, instance_df.shape[0])
    def gen():
        for i in range(num_of_instances):
            yield generate_simple_algorithm_example(instance_df, i, k=0) # currenly set args.incontext_k to 0
    dataset = Dataset.from_generator(generator=gen)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid_dataset = dataset["test"].train_test_split(test_size=0.5, seed=42)
    train_dataset, valid_dataset, test_dataset = dataset["train"], test_valid_dataset["train"], test_valid_dataset["test"]
    if len(train_dataset) > train_size:
        train_dataset = train_dataset.select(range(train_size))
    if len(valid_dataset) > valid_size:
        valid_dataset = valid_dataset.select(range(valid_size))
    if len(test_dataset) > test_size:
        test_dataset = test_dataset.select(range(test_size))
    print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset

def main(config, args):
    # setup logger
    logger = config.get_logger('train')
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()

    # load data
    train_dataset, valid_dataset, test_dataset = load_algorithmic_dataset(
        args.algorithm, args.data_dir, args.train_size, args.valid_size, args.test_size, args.num_of_instances
    )
    if args.load_target:
        target_train_dataset, target_valid_dataset, target_test_dataset = load_algorithmic_dataset(
            args.target_algorithm, args.target_data_dir, args.train_size, args.valid_size, args.test_size, args.num_of_instances
        )

        if args.combine_target:
            train_dataset = datasets.concatenate_datasets([train_dataset, target_train_dataset])
            valid_dataset = datasets.concatenate_datasets([valid_dataset, target_valid_dataset])
            print("Combine source and target dataset", len(train_dataset), len(valid_dataset))

    # initialize model
    model, tokenizer = load_model(args.model_name, not args.random_init, args.num_layers, args.tokenizer_dir)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    split_attention = False

    load_model_dir = os.path.join("saved", args.load_model_dir); state_dict = None
    if os.path.exists(load_model_dir):
        state_dict = torch.load(os.path.join(load_model_dir, "model_best.pth"), 
                                map_location=device)["state_dict"]
        # determine whether to split attention for the state dict 
        for key in list(state_dict.keys()):
            if "weight_k" in key:
                split_attention = True
        if split_attention: 
            for name, module in model.named_modules():
                if "c_attn" in name:
                    split_gpt_self_attention(module, "weight")
        model.load_state_dict(state_dict)
        print("Load model from {}".format(load_model_dir))

    if args.model_interpolate:
        model_start_dir = os.path.join("saved", args.model_interpolate_start)
        model_end_dir = os.path.join("saved", args.model_interpolate_end)
        assert os.path.exists(model_start_dir) and os.path.exists(model_end_dir)
        model_start = torch.load(os.path.join(model_start_dir, "model_best.pth"), map_location=device)["state_dict"]
        model_end = torch.load(os.path.join(model_end_dir, "model_best.pth"), map_location=device)["state_dict"]

        model_inter = interpolate_models(model_start, model_end, args.model_interpolate_alpha)
        model.load_state_dict(model_inter)
        print("Interpolate model from {} and {} with alpha {}".format(model_start_dir, model_end_dir, args.model_interpolate_alpha))

    # if not split_attention in loading check point, split attention now
    if (not split_attention) and (args.train_constraint_allocation or args.train_constraint):
        for name, module in model.named_modules():
            if "c_attn" in name:
                split_gpt_self_attention(module, "weight")

    # initialize data loader
    collator = CLMCollator(tokenizer, max_length=args.max_length, no_position_ids=args.no_position_ids)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    metrics = defaultdict(list)
    source_state_dict = deep_copy(model.state_dict())
    for run in range(args.runs):
        model.load_state_dict(deep_copy(source_state_dict))
        if args.reset_lm_head: # for transferring between algorithms, we reset the lm_head
            model.lm_head.reset_parameters()

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_data_loader) / config["trainer"]["gradient_accumulation_steps"])
        if config["trainer"]["max_train_steps"] == -1: # if max_steps is specified, use max_steps
            config["trainer"]["max_train_steps"] = config["trainer"]["num_train_epochs"] * num_update_steps_per_epoch
        else:
            config["trainer"]["num_train_epochs"] = math.ceil(config["trainer"]["max_train_steps"] / num_update_steps_per_epoch)
        
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=config["trainer"]["num_warmup_steps"]*num_update_steps_per_epoch,
            num_training_steps=config["trainer"]["max_train_steps"]*num_update_steps_per_epoch,
        )

        # Train
        checkpoint_dir = os.path.join("saved", 
            "{}_{}_{}_layers_{}_train_size_{}".format(
                args.model_name, args.algorithm, args.data_dir, args.num_layers, args.train_size
            ) \
            + ("_load_model" if os.path.exists(load_model_dir) else "")\
            + ("_interpolate" if args.model_interpolate else "") \
            + ("_constraint" if args.reg_method == "constraint" else "") \
            + ("_customized_constraint" if args.reg_method == "customized_constraint" else "") \
            + ("_penalty" if args.reg_method == "penalty" else "")\
            + ("_ls" if args.train_ls else "")\
            + ("_joint_training" if args.combine_target else "")\
            + ("_constraint_allocation" if args.train_constraint_allocation else "")\
            + ("_sam" if args.train_sam else "")\
            + ("_mixout" if args.train_mixout else "")
        )
        compute_metrics_for_steps = ("step" in train_dataset.column_names)
        max_inter_steps = max(train_dataset['step']) if compute_metrics_for_steps else 10
        if args.train_constraint:
            # model.register_parameter(name='lm_head', param=model.lm_head.weight)
            trainer = ConstraintTrainer(model, tokenizer, optimizer, lr_scheduler,
                            config=config,
                            device=device,
                            logger=logger,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            checkpoint_dir = checkpoint_dir,
                            generate_length=args.generate_length,
                            compute_metrics_for_steps = compute_metrics_for_steps,
                            max_inter_steps = max_inter_steps
                            )
            if args.reg_method == "constraint":
                trainer.add_constraint(
                    lambda_q = args.reg_attention, 
                    lambda_k=args.reg_attention,
                    lambda_v=args.reg_attention,
                    lambda_linear_1=args.reg_linear, 
                    lambda_linear_2=args.reg_linear,
                    lambda_linear_3=args.reg_linear,
                    state_dict=source_state_dict
                )
            elif args.reg_method == "customized_constraint":
                trainer.add_constraint(
                    lambda_q = args.reg_q, 
                    lambda_k = args.reg_k,
                    lambda_v = args.reg_v,
                    lambda_linear_1=args.reg_linear, 
                    lambda_linear_2=args.reg_linear,
                    lambda_linear_3=args.reg_linear,
                    state_dict=source_state_dict
                )
            if args.reg_method == "penalty":
                trainer.add_penalties(
                    lambda_attention = args.reg_attention, 
                    lambda_linear=args.reg_linear, 
                    lambda_pred_head=args.reg_predictor, 
                    state_dict = source_state_dict
                )
        elif args.train_constraint_allocation:
            trainer = ConstraintAllocationTrainer(model, tokenizer, optimizer, lr_scheduler,
                            config=config,
                            device=device,
                            logger=logger,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            checkpoint_dir = checkpoint_dir,
                            generate_length=args.generate_length,
                            compute_metrics_for_steps = compute_metrics_for_steps,
                            max_inter_steps = max_inter_steps
                            )   
            trainer.add_constraint(
                    constraint_lambda=args.reg_total,
                    state_dict=source_state_dict,
                    allocation_method=args.allocation_strategy
                )                  
        elif args.train_ls:
            trainer = LabelSmoothTrainer(model, tokenizer, optimizer, lr_scheduler,
                            config=config,
                            device=device,
                            logger=logger,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            checkpoint_dir = checkpoint_dir,
                            generate_length=args.generate_length,
                            alpha = args.ls_alpha,
                            num_classes=tokenizer.vocab_size,
                            compute_metrics_for_steps = compute_metrics_for_steps,
                            max_inter_steps = max_inter_steps
                            )
        elif args.train_sam:
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(optimizer_grouped_parameters, base_optimizer, rho=args.sam_rho, adaptive=False, lr=args.lr)
            lr_scheduler = get_scheduler(
                name=args.lr_scheduler_type,
                optimizer=optimizer.base_optimizer,
                num_warmup_steps=config["trainer"]["num_warmup_steps"]*num_update_steps_per_epoch,
                num_training_steps=config["trainer"]["max_train_steps"],
            )
            trainer = SAMTrainer(model, tokenizer, optimizer, lr_scheduler,
                            config=config,
                            device=device,
                            logger=logger,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            checkpoint_dir = checkpoint_dir,
                            generate_length=args.generate_length,
                            compute_metrics_for_steps = compute_metrics_for_steps,
                            max_inter_steps = max_inter_steps
                            )
        elif args.train_mixout:
            # use tuple to avoid OrderedDict warning
            for name, module in tuple(model.transformer.named_modules()):
                if name:
                    recursive_setattr(model.transformer, name, replace_layer_for_mixout(module, mixout_prob=args.mixout_prob))
            print(model)
            trainer = Trainer(model, tokenizer, optimizer, lr_scheduler,
                            config=config,
                            device=device,
                            logger=logger,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            checkpoint_dir = checkpoint_dir,
                            generate_length=args.generate_length,
                            compute_metrics_for_steps = compute_metrics_for_steps,
                            max_inter_steps = max_inter_steps
                            )
        else:
            trainer = Trainer(model, tokenizer, optimizer, lr_scheduler,
                            config=config,
                            device=device,
                            logger=logger,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            checkpoint_dir = checkpoint_dir,
                            generate_length=args.generate_length,
                            compute_metrics_for_steps = compute_metrics_for_steps,
                            max_inter_steps = max_inter_steps
                            )
        log = trainer.train()
        test_log = trainer.test(load_best=(config["trainer"]["max_train_steps"]!=0))
        print(test_log)
        for key, val in test_log.items():
            metrics[key].append(val)
        
        if args.load_target:
            compute_metrics_for_steps = ("step" in target_train_dataset.column_names),
            max_inter_steps = max(target_train_dataset['step']) if compute_metrics_for_steps else 10
            target_train_data_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
            target_valid_data_loader = DataLoader(target_valid_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
            target_test_data_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
            target_config = copy.deepcopy(config)
            target_config["trainer"]["max_train_steps"] = 0
            target_config["trainer"]["num_train_epochs"] = 0
            
            trainer.test(load_best=True) # load best model
            if args.reset_lm_head and state_dict is not None: # if we have reset the lm_head, 
                model.lm_head.weight.data = state_dict["lm_head.weight"].clone() # load the lm_head from the source model
            target_trainer = Trainer(model, tokenizer, optimizer, lr_scheduler,
                            config=target_config,
                            device=device,
                            logger=logger,
                            train_data_loader = target_train_data_loader,
                            valid_data_loader = target_valid_data_loader,
                            test_data_loader = target_test_data_loader,
                            checkpoint_dir = checkpoint_dir,
                            generate_length=args.target_generate_length,
                            save_intial_model = False,
                            compute_metrics_for_steps = compute_metrics_for_steps,
                            max_inter_steps = max_inter_steps
                            )
            target_test_log = target_trainer.test(load_best=False)
            print("Source task test log:")
            print(target_test_log)

            target_test_log = {"target_" + key: val for key, val in target_test_log.items()}
            # compute average
            for key in test_log.keys():
                target_test_log["average_"+key] = (target_test_log["target_"+key] + test_log[key]) / 2

            for key, val in target_test_log.items():
                metrics[key].append(val)


    for key, vals in metrics.items():
        logger.info("{}: {:.4f} +/- {:.4f}".format(key, np.mean(vals), np.std(vals)))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    
    args.add_argument('--model_name', default="gpt2", type=str)
    args.add_argument('--random_init', default=False, action="store_true")
    args.add_argument('--device', default="0", type=str)
    args.add_argument('--num_layers', default=12, type=int)
    args.add_argument('--no_position_ids', default=False, action="store_true")
    args.add_argument('--tokenizer_dir', default="gpt2_sort_100", type=str)
    args.add_argument('--load_model_dir', default="test", type=str)
    args.add_argument('--reset_lm_head', default=False, action="store_true")

    args.add_argument('--algorithm', default="sorting", type=str)
    args.add_argument('--data_dir', default="length_5", type=str)
    args.add_argument('--generate_length', default=5, type=int)
    args.add_argument('--batch_size', default=1024, type=int)
    args.add_argument('--max_length', default=16, type=int)
    args.add_argument('--incontext_k', default=0, type=int)
    args.add_argument('--num_of_instances', default=1000000, type=int)
    args.add_argument('--train_size',  default=10000, type=int)
    args.add_argument('--valid_size',  default=10000, type=int)
    args.add_argument('--test_size',  default=10000, type=int)

    args.add_argument('--lr', default=1e-5, type=float)
    args.add_argument('--weight_decay', default=0.0, type=float)
    args.add_argument('--lr_scheduler_type', default="cosine", type=str)
    args.add_argument('--runs', default=1, type=int)

    # Load target dataset
    args.add_argument('--load_target', default=False, action="store_true")
    args.add_argument('--target_algorithm', default="sorting", type=str)
    args.add_argument('--target_data_dir', default="length_10", type=str)
    args.add_argument('--target_generate_length', default=10, type=int)
    args.add_argument('--combine_target', default=False, action="store_true")

    # Model interpolation
    args.add_argument('--model_interpolate', default=False, action="store_true")
    args.add_argument('--model_interpolate_start', default="test", type=str)
    args.add_argument('--model_interpolate_end', default="test", type=str)
    args.add_argument('--model_interpolate_alpha', default=1, type=float)

    # Label smoothing
    args.add_argument('--train_ls', action="store_true")
    args.add_argument('--ls_alpha', type=float, default=0.15)

    # Sharpness-aware Minimization
    args.add_argument('--train_sam', action="store_true")
    args.add_argument('--sam_rho', type=float, default=0.05)

    # Mixout
    args.add_argument('--train_mixout', action="store_true")
    args.add_argument('--mixout_prob', type=float, default=0.9)

    # Distance regularization
    args.add_argument('--train_constraint', default=False, action="store_true")
    args.add_argument("--reg_method", type=str, default="none")
    args.add_argument("--reg_attention", type=float, default=1.0)
    args.add_argument("--reg_linear", type=float, default=1.0)
    args.add_argument("--reg_predictor", type=float, default=1.0)

    # Constraint allocation
    args.add_argument('--train_constraint_allocation', default=False, action="store_true")
    args.add_argument("--reg_total", type=float, default=12.0)
    args.add_argument("--allocation_strategy", type=str, default="topk")

    # Customized attention
    args.add_argument("--reg_q", type=float, default=1.0)
    args.add_argument("--reg_k", type=float, default=1.0)
    args.add_argument("--reg_v", type=float, default=1.0)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--epochs'], type=int, target='trainer;num_train_epochs'),
        CustomArgs(['--max_steps'], type=float, target='trainer;max_train_steps'),
        CustomArgs(['--warmup_steps'], type=int, target='trainer;num_warmup_steps'),
        CustomArgs(['--early_stop'], type=int, target='trainer;early_stop'),
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args)
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
from data_loader.multitask_dataset import MultitaskDataset, MultitaskBatchSampler, MultitaskCollator
from third_party.models import MultitaskGPT2LMHeadModel

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
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False) 
        else:
            tokenizer = AutoTokenizer.from_pretrained(f"./tokenizers/{tokenizer_dir}", use_fast=True) 
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

def load_multitask_model(model_name, if_pretrained=True, num_layers=12, tokenizer_dir="gpt2_sort_100", tasks=[], share_head=False):
    """
    Return the model and tokenizer
    """
    if model_name in ['gpt2', "gpt2-medium", "gpt2-large", "gpt2-xl",]:
        if if_pretrained:
            model = MultitaskGPT2LMHeadModel.from_pretrained(model_name, tasks=tasks, share_head=share_head
                        # torch_dtype=torch.float16, 
                        #  ignore_mismatched_sizes=True
                        )
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False) 
        else:
            tokenizer = AutoTokenizer.from_pretrained(f"./tokenizers/{tokenizer_dir}", use_fast=True) 
            config = AutoConfig.from_pretrained("gpt2")
            config.n_layer = num_layers
            config.vocab_size = tokenizer.vocab_size
            print("Num layers: {} Vocab size: {}".format(config.n_layer, config.vocab_size))
            model = MultitaskGPT2LMHeadModel._from_config(config, tasks=tasks, share_head=share_head)
            
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

def load_algorithmic_dataset_with_intermediate_steps(algorithm, data_dir, train_size, valid_size, test_size, num_of_instances=1000000, 
                                                     only_ouptut=False, concatenate_steps=False):
    task_to_train_datasets, task_to_valid_datasets, task_to_test_datasets = {}, {}, {}
    
    file_name = f"./data/{algorithm}/{data_dir}.csv"
    instance_df = pd.read_csv(file_name, index_col=0)

    data_len = min(num_of_instances, instance_df.shape[0])
    rng = np.random.default_rng(42)
    shuffle_idxes = rng.permutation(data_len)
    train_idxes = shuffle_idxes[:train_size]
    valid_idxes = shuffle_idxes[-valid_size-test_size:-test_size]
    test_idxes = shuffle_idxes[-test_size:]

    train_df = instance_df.iloc[train_idxes]
    valid_df = instance_df.iloc[valid_idxes]
    test_df = instance_df.iloc[test_idxes]

    def generate_dataset(instance_df, input_columns=[], output_columns=[]):
        num_of_instances = instance_df.shape[0]
        def gen(input_columns, output_columns):
            for i in range(num_of_instances):
                yield generate_simple_algorithm_example(instance_df, i, k=0,
                        input_columns=input_columns, output_columns=output_columns) # currenly set args.incontext_k to 0
        dataset = Dataset.from_generator(
            generator=gen, cache_dir="./data/cache",
            gen_kwargs={"input_columns": input_columns, "output_columns": output_columns})
        return dataset

    column_names = ([f"step_{i}" for i in range(len(instance_df.columns)) if f"step_{i}" in instance_df.columns] + ["output"])\
         if not only_ouptut else ["output"]
    for i, column_name in enumerate(column_names):
        tmp_train_df = train_df[~train_df[column_name].isna()]
        tmp_valid_df = valid_df[~valid_df[column_name].isna()]
        tmp_test_df = test_df[~test_df[column_name].isna()]
        if len(tmp_train_df) == 0 or len(tmp_valid_df) < 100 or len(tmp_test_df) < 100:
            continue
        train_dataset = generate_dataset(tmp_train_df, input_columns=["input"], output_columns=(column_names[:] if concatenate_steps else [column_names[i]]))
        valid_dataset = generate_dataset(tmp_valid_df, input_columns=["input"] + (column_names[:i] if concatenate_steps else []), output_columns=[column_names[i]]) # include all previous steps if concatenate_steps
        test_dataset = generate_dataset(tmp_test_df, input_columns=["input"] + (column_names[:i] if concatenate_steps else []), output_columns=[column_names[i]]) # include all previous steps if concatenate_steps
        print(f"name: {column_name}, train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")
        task_name = column_name

        task_to_train_datasets[task_name] = train_dataset
        task_to_valid_datasets[task_name] = valid_dataset
        task_to_test_datasets[task_name] = test_dataset

    return task_to_train_datasets, task_to_valid_datasets, task_to_test_datasets

def load_multitask_dataloaders(task_to_train_datasets, task_to_valid_datasets, task_to_test_datasets,
                           tokenizer=None, max_length=10, batch_size=256):
    task_to_train_dataloaders = {}
    task_to_valid_dataloaders = {}
    task_to_test_dataloaders = {}
    task_to_collator = {}

    collator = CLMCollator(tokenizer, max_length=max_length)

    for task_name in task_to_train_datasets.keys():
        train_dataset, valid_dataset, test_dataset = task_to_train_datasets[task_name], task_to_valid_datasets[task_name], task_to_test_datasets[task_name]
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)

        task_to_train_dataloaders[task_name] = train_data_loader
        task_to_valid_dataloaders[task_name] = valid_data_loader
        task_to_test_dataloaders[task_name] = test_data_loader
        task_to_collator[task_name] = collator

    multitask_train_dataset = MultitaskDataset(task_to_train_datasets)
    multitask_train_sampler = MultitaskBatchSampler(task_to_train_datasets, batch_size)
    multitask_train_collator = MultitaskCollator(task_to_collator)
    multitask_train_dataloader = DataLoader(
        multitask_train_dataset,
        batch_sampler=multitask_train_sampler,
        collate_fn=multitask_train_collator.collator_fn,
    )
    return multitask_train_dataloader, task_to_train_dataloaders, task_to_valid_dataloaders, task_to_test_dataloaders

def main(config, args):
    # setup logger
    logger = config.get_logger('train')
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()

    # load data
    tokenizer = AutoTokenizer.from_pretrained(f"./tokenizers/{args.tokenizer_dir}", use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    task_to_train_datasets, task_to_valid_datasets, task_to_test_datasets = {}, {}, {}
    for algorithm, data_dir, train_size in zip(args.algorithms, args.data_dirs, args.train_sizes):
        tmp_task_to_train_datasets, tmp_task_to_valid_datasets, tmp_task_to_test_datasets = load_algorithmic_dataset_with_intermediate_steps(
            algorithm, data_dir, train_size, args.valid_size, args.test_size, only_ouptut=args.only_output, concatenate_steps=args.concatenate_steps
        )
        task_to_train_datasets.update({f"{algorithm}_{data_dir}_{task_name}": dataset for task_name, dataset in tmp_task_to_train_datasets.items()})
        task_to_valid_datasets.update({f"{algorithm}_{data_dir}_{task_name}": dataset for task_name, dataset in tmp_task_to_valid_datasets.items()})
        task_to_test_datasets.update({f"{algorithm}_{data_dir}_{task_name}": dataset for task_name, dataset in tmp_task_to_test_datasets.items()})
    for key, dataset in task_to_valid_datasets.items():
        print(key, dataset[0])
    multitask_train_dataloader, task_to_train_loaders, task_to_valid_loaders, task_to_test_loaders = load_multitask_dataloaders(
        task_to_train_datasets, task_to_valid_datasets, task_to_test_datasets,
        tokenizer=tokenizer, max_length=args.max_length, batch_size=args.batch_size,
    )


    # initialize model
    tasks = [task for task in task_to_train_loaders.keys()]
    model, tokenizer = load_multitask_model(args.model_name, not args.random_init, args.num_layers, args.tokenizer_dir, tasks=tasks, share_head=args.concatenate_steps)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    split_attention = False

    load_model_dir = os.path.join("saved", args.load_model_dir); state_dict = None
    if os.path.exists(load_model_dir):
        state_dict = torch.load(os.path.join(load_model_dir, "model_best.pth"), 
                                map_location=device)["state_dict"]
        # determine whether to split attention for the state dict 
        task_heads = []
        for key in list(state_dict.keys()):
            if "weight_k" in key:
                split_attention = True
            if "task_head_dict" in key:
                task_heads.append(key.split(".")[1])
        if split_attention: 
            for name, module in model.named_modules():
                if "c_attn" in name:
                    split_gpt_self_attention(module, "weight")
        model.add_heads(task_heads)
        model.load_state_dict(state_dict, strict=False)
        print("Load model from {}".format(load_model_dir))

    # if not split_attention in loading check point, split attention now
    if (not split_attention) and (args.train_constraint):
        for name, module in model.named_modules():
            if "c_attn" in name:
                split_gpt_self_attention(module, "weight")

    metrics = defaultdict(list)
    source_state_dict = deep_copy(model.state_dict())
    for run in range(args.runs):
        model.load_state_dict(deep_copy(source_state_dict))

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
        num_update_steps_per_epoch = math.ceil(len(multitask_train_dataloader) / config["trainer"]["gradient_accumulation_steps"])
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
                args.model_name, 
                "_".join(args.algorithms), 
                "_".join(args.data_dirs), 
                args.num_layers, 
                "_".join([str(size) for size in args.train_sizes])
            ) \
            + ("_load_model" if os.path.exists(load_model_dir) else "")\
            + ("_constraint" if args.reg_method == "constraint" else "") \
            + ("_only_output" if args.only_output else "") 
        )

        trainer = MultitaskTrainer(model, tokenizer, optimizer, lr_scheduler,
                        config=config,
                        device=device,
                        logger=logger,
                        multitask_train_data_loader=multitask_train_dataloader,
                        train_data_loaders=task_to_train_loaders,
                        valid_data_loaders=task_to_valid_loaders,
                        test_data_loaders=task_to_test_loaders,
                        checkpoint_dir = checkpoint_dir,
                        generate_length=args.generate_length,
                        eval_epoch=args.eval_epoch
                        )
        log = trainer.train()
        test_log = trainer.test(load_best=(config["trainer"]["max_train_steps"]!=0))
        print(test_log)
        for key, val in test_log.items():
            metrics[key].append(val)


        if args.load_target:
            target_train_datasets, target_valid_datasets, target_test_datasets = load_algorithmic_dataset_with_intermediate_steps(
                    args.target_algorithm, args.target_data_dir, args.target_train_size, args.valid_size, args.test_size, only_ouptut=args.only_output
                )
            target_train_datasets = {f"{args.target_algorithm}_{args.target_data_dir}_{task_name}": dataset for task_name, dataset in target_train_datasets.items()}
            target_valid_datasets = {f"{args.target_algorithm}_{args.target_data_dir}_{task_name}": dataset for task_name, dataset in target_valid_datasets.items()}
            target_test_datasets = {f"{args.target_algorithm}_{args.target_data_dir}_{task_name}": dataset for task_name, dataset in target_test_datasets.items()}
            target_multitask_train_dataloader, target_train_loaders, target_valid_loaders, target_test_loaders = load_multitask_dataloaders(
                target_train_datasets, target_valid_datasets, target_test_datasets,
                tokenizer=tokenizer, max_length=args.max_length, batch_size=args.batch_size,
            )
            model.add_heads([task_name for task_name in target_train_loaders.keys()])

            trainer.load_best_checkpoint() # load best model
            target_trainer = MultitaskTrainer(model, tokenizer, optimizer, lr_scheduler,
                            config=config,
                            device=device,
                            logger=logger,
                            multitask_train_data_loader=target_multitask_train_dataloader,
                            train_data_loaders=target_train_loaders,
                            valid_data_loaders=target_valid_loaders,
                            test_data_loaders=target_test_loaders,
                            checkpoint_dir = checkpoint_dir,
                            generate_length=args.target_generate_length,
                        )
            target_test_log = target_trainer.test(load_best=False)
            print("Source task test log:")
            print(target_test_log)

            # compute average
            for key in ["loss", "accuracy", "edit_distance"]:
                target_test_log["target_" +key] = target_test_log.pop(key)
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

    args.add_argument('--train_multiple_algorithm', default=False, action="store_true")
    args.add_argument('--algorithms', type=str, nargs='+', default=["insertion_sort"])
    args.add_argument('--data_dirs', type=str, nargs='+', default=["inter_results_length_5"])
    args.add_argument('--train_sizes', type=int, nargs='+', default=[1000, 2000])
    args.add_argument('--only_output', default=False, action="store_true")
    args.add_argument('--concatenate_steps',  default=False, action="store_true")

    args.add_argument('--generate_length', default=5, type=int)
    args.add_argument('--batch_size', default=1024, type=int)
    args.add_argument('--max_length', default=16, type=int)
    args.add_argument('--incontext_k', default=0, type=int)
    args.add_argument('--num_of_instances', default=1000000, type=int)
    args.add_argument('--valid_size',  default=10000, type=int)
    args.add_argument('--test_size',  default=10000, type=int)
    args.add_argument('--eval_epoch', default=1, type=int)


    args.add_argument('--lr', default=1e-5, type=float)
    args.add_argument('--weight_decay', default=0.0, type=float)
    args.add_argument('--lr_scheduler_type', default="cosine", type=str)
    args.add_argument('--runs', default=1, type=int)

    # Load target dataset
    args.add_argument('--load_target', default=False, action="store_true")
    args.add_argument('--target_algorithm', default="insertion_sort", type=str)
    args.add_argument('--target_data_dir', default="inter_results_length_5", type=str)
    args.add_argument('--target_train_size', default=1000, type=int)
    args.add_argument('--target_generate_length', default=10, type=int)
    args.add_argument('--combine_target', default=False, action="store_true")

    # Distance regularization
    args.add_argument('--train_constraint', default=False, action="store_true")
    args.add_argument("--reg_method", type=str, default="none")
    args.add_argument("--reg_attention", type=float, default=1.0)
    args.add_argument("--reg_linear", type=float, default=1.0)
    args.add_argument("--reg_predictor", type=float, default=1.0)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--epochs'], type=int, target='trainer;num_train_epochs'),
        CustomArgs(['--max_steps'], type=float, target='trainer;max_train_steps'),
        CustomArgs(['--warmup_steps'], type=int, target='trainer;num_warmup_steps'),
        CustomArgs(['--early_stop'], type=int, target='trainer;early_stop'),
        CustomArgs(['--monitor'], type=int, target='trainer;monitor'),
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args)
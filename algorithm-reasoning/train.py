import torch

# print("ckpt 2:")
# if torch.cuda.is_available():
#     print("CUDA is available")
# else: print("CUDA is not available")

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

def load_algorithmic_dataset_with_intermediate_steps(algorithm, data_dir, train_size, valid_size, test_size, num_of_instances=1000000, 
                                                     only_ouptut=False):
    """ Load intermediate steps into a single sequence """    
    file_name = f"./data/{algorithm}/{data_dir}.csv"
    
    print("**************filename: ",file_name)
    
    # instance_df = pd.read_csv(file_name)
    # print(instance_df.head()) 
    
    instance_df = pd.read_csv(file_name, index_col=0)
    print(instance_df.head()) 
    
    data_len = min(num_of_instances, instance_df.shape[0])
    rng = np.random.default_rng(42)
    shuffle_idxes = rng.permutation(data_len)
    train_idxes = shuffle_idxes[:train_size]
    valid_idxes = shuffle_idxes[-valid_size-test_size:-test_size]
    test_idxes = shuffle_idxes[-test_size:]

    train_df = instance_df.iloc[train_idxes]
    valid_df = instance_df.iloc[valid_idxes]
    test_df = instance_df.iloc[test_idxes]

    print("*************train_idxes*************")
    print(train_idxes)

    def generate_dataset(instance_df, input_columns=[], output_columns=[]):
        num_of_instances = instance_df.shape[0]
        def gen(input_columns, output_columns):
            for i in range(num_of_instances):
                yield generate_simple_algorithm_example(instance_df, i, k=0,
                        input_columns=input_columns, output_columns=output_columns) # currenly set args.incontext_k to 0
        dataset = Dataset.from_generator(
            generator=gen, cache_dir="./data/cache_single_prompt",
            gen_kwargs={"input_columns": input_columns, "output_columns": output_columns})
        print("done generate dataset")
        return dataset

    column_names = ([f"step_{i}" for i in range(min(len(instance_df.columns), 30)) if f"step_{i}" in instance_df.columns] + ["output"])\
         if not only_ouptut else ["output"]

    train_dataset = generate_dataset(train_df, input_columns=["input"], output_columns=column_names[:])
    valid_dataset = generate_dataset(valid_df, input_columns=["input"], output_columns=column_names[:]) # include all previous steps if concatenate_steps
    test_dataset = generate_dataset(test_df, input_columns=["input"], output_columns=column_names[:]) # include all previous steps if concatenate_steps
    print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")


    return train_dataset, valid_dataset, test_dataset, column_names

def main(config, args):
    # setup logger

    print("ckpt 1:")
    if torch.cuda.is_available():
        print("CUDA is available")
    else: print("CUDA is not available")

    logger = config.get_logger('train')
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()

    print("********start load data***********")
    # load data
    tokenizer = AutoTokenizer.from_pretrained(f"./tokenizers/{args.tokenizer_dir}", use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    print("*************done load data************")
    # task_to_train_datasets, task_to_valid_datasets, task_to_test_datasets = {}, {}, {}
    for algorithm, data_dir, train_size in zip(args.algorithms, args.data_dirs, args.train_sizes):
        train_dataset, valid_dataset, test_dataset, output_columns = load_algorithmic_dataset_with_intermediate_steps(
            algorithm, data_dir, train_size, args.valid_size, args.test_size, only_ouptut=args.only_output
        )
    print("******************check run****************")

    # initialize data loader
    collator = CLMCollator(tokenizer, max_length=args.max_length, return_indices=True, output_columns=output_columns)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    # initialize model
    model, tokenizer = load_model(args.model_name, not args.random_init, args.num_layers, args.tokenizer_dir)
    print("torch.cuda.is_available: ",torch.cuda.is_available())
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
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
    if (not split_attention):
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
        checkpoint_dir = os.path.join("saved_new", 
            "{}_{}_{}_layers_{}_train_size_{}".format(
                args.model_name, 
                "_".join(args.algorithms), 
                "_".join(args.data_dirs), 
                args.num_layers, 
                "_".join([str(size) for size in args.train_sizes])
            ) \
            + ("_only_output" if args.only_output else "") \
            + ("_load_model" if os.path.exists(load_model_dir) else "")
        )
    
        trainer = SinglePromptMultitaskTrainer(model, tokenizer, optimizer, lr_scheduler,
                        config=config,
                        device=device,
                        logger=logger,
                        train_data_loader=train_data_loader,
                        valid_data_loader=valid_data_loader,
                        test_data_loader=test_data_loader,
                        checkpoint_dir = checkpoint_dir,
                        generate_length=args.generate_length,
                        eval_epoch=args.eval_epoch,
                        output_columns=output_columns
                        )
        log = trainer.train()
        test_log = trainer.test(load_best=(config["trainer"]["max_train_steps"]!=0))
        for key, val in test_log.items():
            metrics[key].append(val)

    for key, vals in metrics.items():
        logger.info("{}: {:.4f} +/- {:.4f}".format(key, np.mean(vals), np.std(vals)))

if __name__ == '__main__':
    print("ckpt 3:")
    if torch.cuda.is_available():
        print("CUDA is available")
    else: print("CUDA is not available")

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

    # Model interpolation
    args.add_argument('--model_interpolate', default=False, action="store_true")
    args.add_argument('--model_interpolate_start', default="test", type=str)
    args.add_argument('--model_interpolate_end', default="test", type=str)
    args.add_argument('--model_interpolate_alpha', default=1, type=float)


    print("ckpt 4:")
    if torch.cuda.is_available():
        print("CUDA is available")
    else: print("CUDA is not available")

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
    print("ckpt 5:")
    if torch.cuda.is_available():
        print("CUDA is available")
    else: print("CUDA is not available")
    main(config, args)
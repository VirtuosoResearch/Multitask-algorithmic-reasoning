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

from utils.hessian import compute_hessian_traces, set_seed, compute_eigenvalue, get_layers
from utils.adjustment import split_gpt_self_attention
from utils import prepare_inputs


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

def main(config, args):
    set_seed(0)
    logger = config.get_logger('train')
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()

    # load data
    train_dataset, valid_dataset, test_dataset = load_algorithmic_dataset(
        args.algorithm, args.data_dir, args.train_size, args.valid_size, args.test_size, args.num_of_instances
    )

    # initialize model
    model, tokenizer = load_model(args.model_name, not args.random_init, args.num_layers, args.tokenizer_dir)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    split_attention = False

    load_model_dir = os.path.join("saved", args.load_model_dir)
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

    # initialize data loader
    collator = CLMCollator(tokenizer, max_length=args.max_length, no_position_ids=args.no_position_ids)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    
    max_traces = []
    sample_count = 0
    
    hessian_traces = []
    hessian_lambdas = []
    model.eval()
    for _, batch in enumerate(train_data_loader):
        batch = prepare_inputs(batch, device)
        outputs = model(**batch)
        loss = outputs.loss

        layer_traces, _ = compute_hessian_traces(model, loss, device = device)
        lambda_1, _ = compute_eigenvalue(model, loss, device=device, top_n=1) 

        hessian_traces.append(layer_traces)
        hessian_lambdas.append(np.array(lambda_1[0]))

        logger.info("Current layer traces: {}".format(layer_traces))
        print("Avg traces: ", np.mean(np.array(hessian_traces), axis=0), np.mean(np.array(hessian_traces), axis=0).sum())
        print("Avg singular values: ", np.mean(np.array(hessian_lambdas), axis=0), np.mean(np.array(hessian_lambdas), axis=0).sum())
        if sample_count == 0:
            max_traces = layer_traces
        else:
            max_traces = np.maximum(max_traces, layer_traces)
        logger.info("Max Traces: {}".format(max_traces))
        logger.info("Max Traces sum: {}".format(np.sum(max_traces)))
        logger.info("========== Batch Complete ==========")

        sample_count += 1
        if sample_count >= args.compute_batchs:
            break
    print("Sum of trace: {}".format(np.mean(np.array(hessian_traces), axis=0).sum()))
    print("Sum of top-1 eigenvalues: {}".format(np.mean(np.array(hessian_lambdas), axis=0).sum()))


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

    # samples used for computing hessian
    args.add_argument("--compute_batchs", type=int, default=100)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--epochs'], type=int, target='trainer;num_train_epochs'),
        CustomArgs(['--max_steps'], type=float, target='trainer;max_train_steps'),
        CustomArgs(['--warmup_steps'], type=int, target='trainer;num_warmup_steps'),
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args)
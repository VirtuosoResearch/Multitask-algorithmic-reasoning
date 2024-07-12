import argparse
import json
import os
import time

import pandas as pd
import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from compute_metrics import compute_metrics_for_sorting

algortihm_to_prompt = {
    "insertion_sort": "Insertion sort: ",
}

def process_intermediate_steps(intermediate_steps, is_array=True, is_graph=False):
    '''
    Break intermediate steps into a list
    '''
    if is_array:
        intermediate_steps =  intermediate_steps.strip("[]").split("], [")
        intermediate_steps = [f"Array: [" + step + "]" for i, step in enumerate(intermediate_steps)]
        return intermediate_steps
    elif is_graph:
        # TODO: process graph
        pass

def process_intermediate_variables(intermediate_variables):
    intermediate_variables = eval(intermediate_variables)
    intermediate_variables = ["Compare index {} and {}, Swap: {},".format(step["i"], step["j"], step["swap"]) for i, step in enumerate(intermediate_variables)]
    return intermediate_variables

def format_algorithm_process(df, idx):
    process_prompt = ""
    intermediate_variables = df["intermediate_variables"].iloc[idx]
    intermediate_variables = process_intermediate_variables(intermediate_variables)

    intermediate_steps = df["intermediate_steps"].iloc[idx]
    intermediate_steps = process_intermediate_steps(intermediate_steps, is_array=True, is_graph=False)

    for i, step in enumerate(intermediate_steps):
        intermediate_steps[i] = str(intermediate_variables[i]) + " " + step 
    
    for i, step in enumerate(intermediate_steps):
        process_prompt += f"\n" + step
    return process_prompt

def format_algorithm_example(df, idx, 
        init_prompt="Sort the array with insertion sort", 
        include_input=True,
        include_answer=True, 
        include_inter_results=True):
    prompt = init_prompt
    if include_input:
        prompt += df["input"].iloc[idx]

    if include_inter_results:
        process_prompt = format_algorithm_process(df, idx)
        prompt += process_prompt

    if include_answer:
        prompt += "\nAnswer:"
        output = df["output"].iloc[idx]
        prompt += " {}\n\n".format(output)
    return prompt

def generate_algorithm_prompt(train_df, k=-1, algorithm_name="insertion sort", init_prompt="Sort the array with insertion sort"):
    prompt = f"The following are examples of excuting {algorithm_name} algorithm.\n\n"
    import numpy as np
    indexes = np.random.choice(train_df.shape[0], size=k, replace=False)
    for i in indexes:
        prompt += format_algorithm_example(train_df, i, init_prompt=init_prompt)
    return prompt

def load_model(model_name):
    """
    Return the model and tokenizer
    """
    if model_name in ['gpt2', "gpt2-medium", "gpt2-large", "gpt2-xl",]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, 
                                                    #  ignore_mismatched_sizes=True, n_positions=
                                                     )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

    return model, tokenizer

def prepare_input(tokenizer, prompts, device):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)

    return input_tokens

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts, device, max_new_tokens=128):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input, device)
        outputs = model.generate(**encode_inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        # remove the given prompt in the output
        input_len = encode_inputs["input_ids"].shape[1]
        outputs[:, :input_len][encode_inputs["attention_mask"].bool()] = tokenizer.pad_token_id
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers.extend(outputs)
    # answers = [answer for answer in answers]
    return answers

def main(args):
    run_results = {}

    # load algorithm execution examples as pandas dataframe
    file_name = f"./data/{args.algorithm}/{args.data_dir}.csv"
    instance_df = pd.read_csv(file_name, index_col=0)
    num_of_instances = instance_df.shape[0]
    
    # load model and tokenizer
    model_name = args.model_name
    model, tokenizer = load_model(model_name)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    checkpoint_dir = os.path.join(args.checkpoint_dir, "model_best.pth")
    if os.path.exists(checkpoint_dir):
        model.load_state_dict(torch.load(checkpoint_dir, map_location=device)["state_dict"])
        print("Load model from {}".format(checkpoint_dir))
    # generate prompts
    records = []
    k = args.num_train
    for i in range(args.num_examples):
        # get prompt: first select one test example
        prompt_end = format_algorithm_example(instance_df, i, 
                        init_prompt=algortihm_to_prompt[args.algorithm],
                        include_answer=False, include_inter_results=False)
        # generate in-context training examples
        train_prompt = generate_algorithm_prompt(instance_df, k, 
                        algorithm_name=args.algorithm, 
                        init_prompt=algortihm_to_prompt[args.algorithm])
        prompt = train_prompt + prompt_end
        while len(tokenizer.tokenize(prompt)) + 1 > 2048: # bos token
            prompt_split = prompt.split("\n\n")
            prompt_split.pop(1)
            prompt = '\n\n'.join(prompt_split)
        label = format_algorithm_example(instance_df, i, init_prompt="", include_input=False)
        records.append({'prompt':prompt, 'answer':label})

    # evaluate model
    pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records], device, max_new_tokens=args.max_gen_tokens)
    gold_answers = [[record['answer']] for record in records]
    for i, answer in enumerate(pred_answers):
        print("Prompt: =====================================")
        print(records[i]['prompt'])
        print("Generated answer: =====================================")
        print(answer.replace("\n\n", "\n"))
        print("True answer: =====================================")
        print(gold_answers[i][0])

    metrics = compute_metrics_for_sorting(pred_answers, gold_answers, length=args.length)
    print("Exact match: {:.2f}%".format(metrics["exact_match"]))
    print("Compare accuracy: {:.2f}%".format(metrics["compare_accuracy"]))
    print("Swap accuracy: {:.2f}%".format(metrics["swap_accuracy"]))
    print("Intermediate result accuracy: {:.2f}%".format(metrics["inter_accuracy"]))
    print("Output answer accuracy: {:.2f}%".format(metrics["answer_accuracy"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name', type=str, default="gpt2")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--max_gen_tokens', type=int, default=256)
    parser.add_argument('--checkpoint_dir', type=str, default="saved/gpt2_insertion_sort_length_5")

    # data
    parser.add_argument('--algorithm', type=str, default='insertion_sort')
    parser.add_argument('--data_dir', type=str, default='length_5', help="name of the csv file")
    parser.add_argument('--length', type=int, default=5, help="length of the array")
    
    # prompt
    parser.add_argument('--num_examples', type=int, default=8, help="number of examples tot test")
    parser.add_argument('--num_train', type=int, default=1, help="number of in-context examples")
    args = parser.parse_args()
    
    main(args)

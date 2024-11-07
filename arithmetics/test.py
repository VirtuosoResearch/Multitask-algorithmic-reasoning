# %%
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from data_loader.utils import format_algorithm_example, generate_simple_algorithm_example
from datasets import Dataset

class args:
    model_name = "gpt2"

    algorithm = "insertion_sort"
    data_dir = "inter_results_length_5"
    incontext_k = 0

# load data
file_name = f"./data/{args.algorithm}/{args.data_dir}.csv"
instance_df = pd.read_csv(file_name, index_col=0) # compression="zip"

# %%
num_of_instances = instance_df.shape[0]

def gen():
    for i in range(num_of_instances):
        yield generate_simple_algorithm_example(instance_df, i, k=args.incontext_k, input_columns=["input"], output_columns=["step_0", "step_1", "step_2"])
        # yield format_algorithm_example(instance_df, i, include_input=True, include_inter_results=True, include_answer=True)
train_dataset = Dataset.from_generator(generator=gen)
# %%
train_dataset = train_dataset.train_test_split(test_size=0.1)
train_dataset, valid_dataset = train_dataset["train"], train_dataset["test"]
# %%
from data_loader.collators import CLMCollator
# tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, padding_side="left")
# model = AutoModelForCausalLM.from_pretrained(args.model_name, 
#                                             #  torch_dtype=torch.float16, 
#                                             #  ignore_mismatched_sizes=True, n_positions=
#                                                 )
tokenizer = AutoTokenizer.from_pretrained("./tokenizers/gpt2_division", use_fast=True)
config = AutoConfig.from_pretrained("gpt2")
config.n_layer = 2
config.vocab_size = tokenizer.vocab_size
model = AutoModelForCausalLM.from_config(config)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = 0

data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, collate_fn=CLMCollator(tokenizer, max_length=16))
model.load_state_dict(torch.load("./saved/gpt2_division_digit_3_layers_2_train_size_1000000/model_best.pth")["state_dict"])

# %%
import torch

model_init = torch.load("./saved/gpt2_addition_digit_10_carry_True_layers_2_train_size_20000_load_model/model_epoch_0.pth")["state_dict"]
model_final = torch.load("./saved/gpt2_addition_digit_10_carry_True_layers_2_train_size_20000_load_model/model_best.pth")["state_dict"]

# %%
for key in model_init.keys():
    if "Bool" in model_init[key].type():
        continue
    if "weight" in key:
        if "c_attn" in key:
            weight_init   = model_init[key]
            weight_final  = model_final[key]
            embedding_dim = weight_init.shape[0]
            print(key + "_q", torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
            print(key + "_k", torch.linalg.norm(weight_init[:, embedding_dim:2*embedding_dim] - weight_final[:, embedding_dim:2*embedding_dim]).item())
            print(key + "_v", torch.linalg.norm(weight_init[:, 2*embedding_dim:3*embedding_dim] - weight_final[:, 2*embedding_dim:3*embedding_dim]).item())
        else:
            print(key, torch.linalg.norm(model_init[key] - model_final[key]).item())
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

# %%
from data_loader.collators import CLMCollator

tokenizer = AutoTokenizer.from_pretrained("./tokenizers/gpt2_arithmetic", use_fast=True)
config = AutoConfig.from_pretrained("gpt2")
config.n_layer = 6
config.vocab_size = tokenizer.vocab_size
model = AutoModelForCausalLM.from_config(config)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = 0


# %%
from utils.adjustment import split_gpt_self_attention
from collections import defaultdict

model_init = torch.load(f"./saved/gpt2_addition_digit_5_carry_True_layers_6_train_size_10000/model_best.pth", map_location=device)["state_dict"]
model.load_state_dict(model_init, strict=False)
for name, module in model.named_modules():
    if "c_attn" in name:
        split_gpt_self_attention(module, "weight")
model_init = model.state_dict()
model_final = torch.load(f"./saved/gpt2_subtraction_digit_5_borrow_True_layers_6_train_size_5000_load_model_constraint_allocation_topk_0.01_30.0/model_best.pth", map_location=device)["state_dict"]
topk_distances = defaultdict(list)

for key in model_init.keys():
    if "Bool" in model_init[key].type():
        continue
    if "weight" in key:
        if "c_attn" in key:
            weight_init   = model_init[key]
            weight_final  = model_final[key]
            embedding_dim = weight_init.shape[0]
            

            
            if "weight_q" in key:
                topk_distances["q"].append(torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
            elif "weight_k" in key:
                topk_distances["k"].append(torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
            elif "weight_v" in key:
                topk_distances["v"].append(torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
        else:
            # print(key, torch.linalg.norm(model_init[key] - model_final[key]).item())
            if "attn.c_proj" in key: 
                topk_distances["o"].append(torch.linalg.norm(model_init[key] - model_final[key]).item())
            if "mlp.c_fc" in key: 
                topk_distances["c_fc"].append(torch.linalg.norm(model_init[key] - model_final[key]).item())
            if "mlp.c_proj" in key:
                topk_distances["c_proj"].append(torch.linalg.norm(model_init[key] - model_final[key]).item())


for key in topk_distances.keys():
    print(key, topk_distances[key])

# %%
from utils.adjustment import split_gpt_self_attention
from collections import defaultdict

# model_init = torch.load(f"./saved/gpt2_subtraction_digit_5_borrow_True_layers_6_train_size_5000_load_model_constraint_allocation_uniform_0.01_35.0/model_epoch_0.pth", map_location=device)["state_dict"]
model_final = torch.load(f"./saved/gpt2_subtraction_digit_5_borrow_True_layers_6_train_size_5000_load_model_constraint_allocation_uniform_0.01_30.0/model_best.pth", map_location=device)["state_dict"]
uniform_distances = defaultdict(list)

for key in model_init.keys():
    if "Bool" in model_init[key].type():
        continue
    if "weight" in key:
        if "c_attn" in key:
            weight_init   = model_init[key]
            weight_final  = model_final[key]
            embedding_dim = weight_init.shape[0]
            
            if "weight_q" in key:
                uniform_distances["q"].append(torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
            elif "weight_k" in key:
                uniform_distances["k"].append(torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
            elif "weight_v" in key:
                uniform_distances["v"].append(torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
        else:
            # print(key, torch.linalg.norm(model_init[key] - model_final[key]).item())
            if "attn.c_proj" in key: 
                uniform_distances["o"].append(torch.linalg.norm(model_init[key] - model_final[key]).item())
            if "mlp.c_fc" in key: 
                uniform_distances["c_fc"].append(torch.linalg.norm(model_init[key] - model_final[key]).item())
            if "mlp.c_proj" in key:
                uniform_distances["c_proj"].append(torch.linalg.norm(model_init[key] - model_final[key]).item())
uniform_distances

for key in uniform_distances.keys():
    print(key, uniform_distances[key])


# %%
import matplotlib.pyplot as plt

key = "q"
plt.plot(topk_distances[key], label="topk")
plt.plot(uniform_distances[key], label="uniform")
plt.title(key)
plt.legend()

# %%
key = "k"
plt.plot(topk_distances[key], label="topk")
plt.plot(uniform_distances[key], label="uniform")
plt.title(key)
plt.legend()

# %%
key = "v"
plt.plot(topk_distances[key], label="topk")
plt.plot(uniform_distances[key], label="uniform")
plt.title(key)
plt.legend()

# %%
key = "o"
plt.plot(topk_distances[key], label="topk")
plt.plot(uniform_distances[key], label="uniform")
plt.title(key)
plt.legend()

# %%
key = "c_fc"
plt.plot(topk_distances[key], label="topk")
plt.plot(uniform_distances[key], label="uniform")
plt.title(key)
plt.legend()

# %%
key = "c_proj"
plt.plot(topk_distances[key], label="topk")
plt.plot(uniform_distances[key], label="uniform")
plt.title(key)

plt.legend()
# %%

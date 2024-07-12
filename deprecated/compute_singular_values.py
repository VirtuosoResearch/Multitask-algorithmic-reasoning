# %%
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from data_loader.utils import format_algorithm_example, generate_simple_algorithm_example
from datasets import Dataset

class args:
    model_name = "gpt2"

    algorithm = "sorting"
    data_dir = "length_10"
    incontext_k = 0

# load data
file_name = f"./data/{args.algorithm}/{args.data_dir}.csv"
instance_df = pd.read_csv(file_name, index_col=0) # compression="zip"

num_of_instances = instance_df.shape[0]

def gen():
    for i in range(num_of_instances):
        yield generate_simple_algorithm_example(instance_df, i, k=args.incontext_k)
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
tokenizer = AutoTokenizer.from_pretrained("./tokenizers/gpt2_sort_100", use_fast=True)
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

# %%
from utils.adjustment import split_gpt_self_attention

model_init = torch.load(f"./saved/gpt2_sorting_length_10_layers_2_train_size_50000_load_model/model_epoch_0.pth")["state_dict"]
model_final = torch.load(f"./saved/gpt2_sorting_length_10_layers_2_train_size_50000_load_model/model_best.pth")["state_dict"]
# for name, module in model.named_modules():
#     if "c_attn" in name:
#         split_gpt_self_attention(module, "weight")

model.load_state_dict(model_final)

# %%
''' Compute the singular values '''
from collections import defaultdict
finetuned_distance = defaultdict(list)
for num in [50000]:
    model_init = torch.load(f"./saved/gpt2_sorting_length_10_layers_2_train_size_{num}_load_model/model_epoch_0.pth")["state_dict"]
    model_final = model.state_dict()
    for key in model_init.keys():
        if "Bool" in model_init[key].type():
            continue
        if "weight" in key:
            if "c_attn" in key:
                weight_init   = model_init[key]
                weight_final  = model_final[key]
                embedding_dim = weight_init.shape[0]
                
                finetuned_distance[key + "_q"] = weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]
                finetuned_distance[key + "_k"] = weight_init[:, embedding_dim:2*embedding_dim] - weight_final[:, embedding_dim:2*embedding_dim]
                finetuned_distance[key + "_v"] = weight_init[:, 2*embedding_dim:3*embedding_dim] - weight_final[:, 2*embedding_dim:3*embedding_dim]
            else:
                # print(key, torch.linalg.norm(model_init[key] - model_final[key]).item())
                if "c_proj" in key or "c_fc" in key:
                    finetuned_distance[key] = model_init[key] - model_final[key]

name = "transformer.h.0.attn.c_attn.weight_k"
for key, value in finetuned_distance.items():
    if name in key:
        U, S, VT = torch.linalg.svd(value)

import matplotlib.pyplot as plt
S = S.cpu().numpy()
plt.plot(S)

S_finetuned = (S.copy())


# %%
from utils import prepare_inputs
import torch
from torch.nn import BatchNorm2d

model_init = torch.load(f"./saved/gpt2_sorting_length_10_layers_2_train_size_50000_load_model/model_epoch_0.pth")["state_dict"]
model_final = torch.load(f"./saved/gpt2_sorting_length_10_layers_2_train_size_50000_load_model/model_best.pth")["state_dict"]

model.load_state_dict(model_final)

def _frob_norm(w):
    return torch.sum(torch.pow(w, 2.0))

def _linf_norm(w):
    if len(w.shape) != 2 and len(w.shape) != 4:
        assert len(w.shape) == 1
        return torch.max(torch.abs(w))
    
    axes=1

    if len(w.shape) == 4:
        axes=[1, 2, 3]
    
    norm = torch.max(torch.sum(torch.abs(w), dim=axes))
    return norm

class TopKFrobeniusConstraint(object): 

    def __init__(self, model_type, max_k, state_dict = None,
                 excluding_key = [], including_key = []) -> None:
        self.model_type = model_type
        self.max_k = max_k
        self.state_dict = state_dict
        self.excluding_key = excluding_key
        self.including_key = including_key

    def __call__(self, module):
        if type(module) == self.model_type:
            param_dict = {}
            score_dict = {}
            for name, param in module.named_parameters():
                if "bias" in name:
                    continue
            
                if (len(self.excluding_key)>0):
                    for key in self.excluding_key:
                        if key in name:
                            continue
                
                for key in self.including_key:
                    if key in name:
                        print(name)
                        param_dict[name] = param

                        w = param.data
                        grad = param.grad.data

                        scores = torch.sign(w*grad)*torch.abs(grad)
                        score_dict[name] = scores
                        break

            # distribute constraint budget according to scores         
            total_score = torch.cat([torch.flatten(v) for v in score_dict.values()])
            total_param = torch.cat([torch.flatten(v.data - self.state_dict[key]) for key, v in param_dict.items()])
            current_norm =  _frob_norm(total_param)
            print(current_norm)
            if current_norm < self.max_k: return

            k = int(len(total_param)*0.01); interval = int(len(total_param)*0.005)
            _, indices = torch.topk(total_score, k, sorted=False)
            while _frob_norm(total_param[indices]) < current_norm - self.max_k:
                k += interval
                _, indices = torch.topk(total_score, k, sorted=False)
            k -= interval
            _, indices = torch.topk(total_score, k, sorted=True)
            print(_frob_norm(total_param[indices]))

            # make the params correspond to indices to zero
            masks = torch.zeros_like(total_param, dtype=torch.bool)
            masks[indices] = 1
            mask_dict = {}; cur_len = 0
            for name, param in param_dict.items():
                tmp_mask = masks[cur_len:cur_len+param.numel()].reshape(param.shape)
                mask_dict[name] = tmp_mask
                cur_len += param.numel()
            
            # apply mask
            for name, param in param_dict.items():
                tmp_mask = mask_dict[name]
                param.data[tmp_mask] = self.state_dict[name][tmp_mask]

constraint = TopKFrobeniusConstraint(type(model), 60, state_dict=model_init, including_key=["c_attn.weight",  "attn.c_proj", "mlp.c_fc", "mlp.c_proj"])

for batch in data_loader:
    batch = prepare_inputs(batch, device)
    output = model(**batch)
    loss = output.loss
    print(loss.item())
    loss.backward()

    model.apply(constraint)

    break

''' Compute the singular values '''
from collections import defaultdict
finetuned_distance = defaultdict(list)
for num in [50000]:
    model_init = torch.load(f"./saved/gpt2_sorting_length_10_layers_2_train_size_{num}_load_model/model_epoch_0.pth")["state_dict"]
    model_final = model.state_dict()
    for key in model_init.keys():
        if "Bool" in model_init[key].type():
            continue
        if "weight" in key:
            if "c_attn" in key:
                weight_init   = model_init[key]
                weight_final  = model_final[key]
                embedding_dim = weight_init.shape[0]
                
                finetuned_distance[key + "_q"] = weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]
                finetuned_distance[key + "_k"] = weight_init[:, embedding_dim:2*embedding_dim] - weight_final[:, embedding_dim:2*embedding_dim]
                finetuned_distance[key + "_v"] = weight_init[:, 2*embedding_dim:3*embedding_dim] - weight_final[:, 2*embedding_dim:3*embedding_dim]
            else:
                # print(key, torch.linalg.norm(model_init[key] - model_final[key]).item())
                if "c_proj" in key or "c_fc" in key:
                    finetuned_distance[key] = model_init[key] - model_final[key]

for key, value in finetuned_distance.items():
    if name in key:
        U, S, VT = torch.linalg.svd(value)

S_topk = S.cpu().numpy().copy()

plt.plot(S_finetuned, label="finetuned")
plt.plot(S_topk, label="topk")
plt.legend()

# %%
from utils.adjustment import split_gpt_self_attention

model_init = torch.load(f"./saved/gpt2_sorting_length_10_layers_2_train_size_50000_load_model/model_epoch_0.pth")["state_dict"]
model_final = torch.load(f"./saved/gpt2_sorting_length_10_layers_2_train_size_50000_load_model/model_best.pth")["state_dict"]

model.load_state_dict(model_final)

class UniformFrobeniusConstraint(object): 

    def __init__(self, model_type, max_k, state_dict = None,
                 excluding_key = [], including_key = []) -> None:
        self.model_type = model_type
        self.max_k = max_k
        self.state_dict = state_dict
        self.excluding_key = excluding_key
        self.including_key = including_key

    def __call__(self, module):
        if type(module) == self.model_type:
            param_dict = {}
            score_dict = {}
            for name, param in module.named_parameters():
                if "bias" in name:
                    continue
            
                if (len(self.excluding_key)>0):
                    for key in self.excluding_key:
                        if key in name:
                            continue
                
                for key in self.including_key:
                    if key in name:
                        param_dict[name] = param

                        w = param.data
                        break

            # distribute constraint budget according to scores         
            total_param = torch.cat([torch.flatten(v.data - self.state_dict[key]) for key, v in param_dict.items()])
            current_norm =  _frob_norm(total_param)
            if current_norm < self.max_k: return

            # rescale proportionally to current norm/max norm
            for name, param in param_dict.items():
                param.data = (param.data - self.state_dict[name]) * (self.max_k/current_norm)**0.5 + self.state_dict[name]

constraint = UniformFrobeniusConstraint(type(model), 60, state_dict=model_init, including_key=["c_attn.weight",  "attn.c_proj", "mlp.c_fc", "mlp.c_proj"])

for batch in data_loader:
    batch = prepare_inputs(batch, device)
    output = model(**batch)
    loss = output.loss
    print(loss.item())
    loss.backward()

    model.apply(constraint)

    break


''' Compute the singular values '''
from collections import defaultdict
finetuned_distance = defaultdict(list)
for num in [50000]:
    model_init = torch.load(f"./saved/gpt2_sorting_length_10_layers_2_train_size_{num}_load_model/model_epoch_0.pth")["state_dict"]
    model_final = model.state_dict()
    for key in model_init.keys():
        if "Bool" in model_init[key].type():
            continue
        if "weight" in key:
            if "c_attn" in key:
                weight_init   = model_init[key]
                weight_final  = model_final[key]
                embedding_dim = weight_init.shape[0]
                
                finetuned_distance[key + "_q"] = weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]
                finetuned_distance[key + "_k"] = weight_init[:, embedding_dim:2*embedding_dim] - weight_final[:, embedding_dim:2*embedding_dim]
                finetuned_distance[key + "_v"] = weight_init[:, 2*embedding_dim:3*embedding_dim] - weight_final[:, 2*embedding_dim:3*embedding_dim]
            else:
                # print(key, torch.linalg.norm(model_init[key] - model_final[key]).item())
                if "c_proj" in key or "c_fc" in key:
                    finetuned_distance[key] = model_init[key] - model_final[key]


for key, value in finetuned_distance.items():
    if name in key:
        U, S, VT = torch.linalg.svd(value)


S_uniform = S.cpu().numpy().copy()

plt.plot(S_topk[:150], label="topk")
plt.plot(S_uniform[:150], label="uniform")
plt.legend()


# %%
import numpy as np
# plt.plot(S_finetuned, label="finetuned")
plt.plot(S_topk[:200], label="topk")
plt.plot(S_uniform[:200], label="uniform")
plt.legend()













# %%
from collections import defaultdict
finetuned_distance = defaultdict(list)
for num in [50000]:
    model_init = torch.load(f"./saved/gpt2_addition_digit_10_carry_True_layers_2_train_size_20000_load_model_constraint/model_epoch_0.pth")["state_dict"]
    model_final = torch.load(f"./saved/gpt2_addition_digit_10_carry_True_layers_2_train_size_20000_load_model_constraint/model_best.pth")["state_dict"]
    for key in model_init.keys():
        if "Bool" in model_init[key].type():
            continue
        if "weight" in key:
            if "c_attn" in key:
                weight_init   = model_init[key]
                weight_final  = model_final[key]
                embedding_dim = weight_init.shape[0]
                
                # print(key + "_q", torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
                # print(key + "_k", torch.linalg.norm(weight_init[:, embedding_dim:2*embedding_dim] - weight_final[:, embedding_dim:2*embedding_dim]).item())
                # print(key + "_v", torch.linalg.norm(weight_init[:, 2*embedding_dim:3*embedding_dim] - weight_final[:, 2*embedding_dim:3*embedding_dim]).item())
                finetuned_distance[key + "_q"].append(torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
                finetuned_distance[key + "_k"].append(torch.linalg.norm(weight_init[:, embedding_dim:2*embedding_dim] - weight_final[:, embedding_dim:2*embedding_dim]).item())
                finetuned_distance[key + "_v"].append(torch.linalg.norm(weight_init[:, 2*embedding_dim:3*embedding_dim] - weight_final[:, 2*embedding_dim:3*embedding_dim]).item())
            else:
                # print(key, torch.linalg.norm(model_init[key] - model_final[key]).item())
                if "c_proj" in key or "c_fc" in key:
                    finetuned_distance[key].append(torch.linalg.norm(model_init[key] - model_final[key]).item())
finetuned_distance
# %%
from collections import defaultdict
finetuned_distance = defaultdict(list)

model_init = torch.load(f"./saved/gpt2_addition_digit_10_carry_True_layers_2_train_size_20000_load_model_constraint/model_epoch_0.pth")["state_dict"]
model_final = torch.load(f"./saved/gpt2_addition_digit_10_carry_True_layers_2_train_size_20000_load_model_constraint/model_best.pth")["state_dict"]
for key in model_init.keys():
    if "Bool" in model_init[key].type():
        continue
    if "weight" in key:
        if "c_attn" in key:
            weight_init   = model_init[key]
            weight_final  = model_final[key]
            embedding_dim = weight_init.shape[0]
            
            # print(key + "_q", torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
            # print(key + "_k", torch.linalg.norm(weight_init[:, embedding_dim:2*embedding_dim] - weight_final[:, embedding_dim:2*embedding_dim]).item())
            # print(key + "_v", torch.linalg.norm(weight_init[:, 2*embedding_dim:3*embedding_dim] - weight_final[:, 2*embedding_dim:3*embedding_dim]).item())
            finetuned_distance[key].append(torch.linalg.norm(weight_init[:, :embedding_dim] - weight_final[:, :embedding_dim]).item())
        else:
            # print(key, torch.linalg.norm(model_init[key] - model_final[key]).item())
            if "c_proj" in key or "c_fc" in key:
                finetuned_distance[key].append(torch.linalg.norm(model_init[key] - model_final[key]).item())
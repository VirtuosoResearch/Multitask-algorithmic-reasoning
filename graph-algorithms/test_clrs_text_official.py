# %%
from datasets import load_dataset

train_dataset = load_dataset("tomg-group-umd/CLRS-Text-train")['train']

# %%
algorithm = 'bellman_ford'
dataset = train_dataset.filter(lambda x: x['algo_name'] == algorithm)

# %%
import numpy as np

def get_length(x):
    question = x['question']
    start_index = question.find('A:')
    end_index = question.find(', initial_trace:')
    question = question[start_index:end_index]
    return len(question.split(','))

lengths = np.array([get_length(x) for x in dataset])
print(len(dataset))
print(np.min(lengths), np.max(lengths))


# %%
import numpy as np

lengths = np.array([x['length'] for x in dataset])
print(len(dataset))
print(np.min(lengths), np.max(lengths))

# min 4 max 28
# min 4 max 60
# min 4 max 46

# %%
datasets =  dataset.train_test_split(test_size=0.1)

# %%
print(dataset[0]['question'])
# %%
# TODO:
# 1. Build a data set for each algorithm
# 2. Conver the input to the graph format input 

# %%
from src.custom.clrs_text_task_data_module import TextCLRSDataModule
from src.custom.clrs_text_task_graph_data_module import TextGraphCLRSDataModule
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.mask_token_id = 128002 # special token
task_name = "bfs"

data_module = TextGraphCLRSDataModule(
    task_names=[task_name],
    tokenizer=tokenizer,
    batch_size=1,
    inference_batch_size=8,
    max_input_length=128,
    max_output_length=128,
    train_lengths=[15],
    test_lengths=[15],
    eval_split=0.3,
    eval_all=True,
    use_few_shot=False,
    few_shot_k=0,
    load_only_last_output=True)
data_module.setup(stage="fit")

# for i in range(30):
#     print(i, tokenizer(f"{i}"))
# 0 {'input_ids': [128000, 15], 'attention_mask': [1, 1]}
# 1 {'input_ids': [128000, 16], 'attention_mask': [1, 1]}
# 2 {'input_ids': [128000, 17], 'attention_mask': [1, 1]}
# 3 {'input_ids': [128000, 18], 'attention_mask': [1, 1]}
# 4 {'input_ids': [128000, 19], 'attention_mask': [1, 1]}
# 5 {'input_ids': [128000, 20], 'attention_mask': [1, 1]}
# 6 {'input_ids': [128000, 21], 'attention_mask': [1, 1]}
# 7 {'input_ids': [128000, 22], 'attention_mask': [1, 1]}
# 8 {'input_ids': [128000, 23], 'attention_mask': [1, 1]}
# 9 {'input_ids': [128000, 24], 'attention_mask': [1, 1]}
# 10 {'input_ids': [128000, 605], 'attention_mask': [1, 1]}
# 11 {'input_ids': [128000, 806], 'attention_mask': [1, 1]}
# 12 {'input_ids': [128000, 717], 'attention_mask': [1, 1]}
# 13 {'input_ids': [128000, 1032], 'attention_mask': [1, 1]}
# 14 {'input_ids': [128000, 975], 'attention_mask': [1, 1]}
# 15 {'input_ids': [128000, 868], 'attention_mask': [1, 1]}
# 16 {'input_ids': [128000, 845], 'attention_mask': [1, 1]}
# 17 {'input_ids': [128000, 1114], 'attention_mask': [1, 1]}
# 18 {'input_ids': [128000, 972], 'attention_mask': [1, 1]}
# 19 {'input_ids': [128000, 777], 'attention_mask': [1, 1]}
# 20 {'input_ids': [128000, 508], 'attention_mask': [1, 1]}
# 21 {'input_ids': [128000, 1691], 'attention_mask': [1, 1]}
# 22 {'input_ids': [128000, 1313], 'attention_mask': [1, 1]}
# 23 {'input_ids': [128000, 1419], 'attention_mask': [1, 1]}
# 24 {'input_ids': [128000, 1187], 'attention_mask': [1, 1]}
# 25 {'input_ids': [128000, 914], 'attention_mask': [1, 1]}
# 26 {'input_ids': [128000, 1627], 'attention_mask': [1, 1]}
# 27 {'input_ids': [128000, 1544], 'attention_mask': [1, 1]}
# 28 {'input_ids': [128000, 1591], 'attention_mask': [1, 1]}
# 29 {'input_ids': [128000, 1682], 'attention_mask': [1, 1]}
# # %%
# import torch
# checkpoint = torch.load("./external_lightning_logs/meta-llama-Llama-3.2-1B_bfs_test_pretraining_cross_attn_run_0/epoch_epoch=19.pt")

# %%
max_length = 0; max_output_length = 0
for test_sample in data_module.task_to_train_datasets[task_name]:
    print(test_sample['input'], test_sample['output'])
    length = tokenizer(test_sample['input'], return_tensors='pt')['input_ids'].shape
    print(length)
    output_length = tokenizer(test_sample['output'], return_tensors='pt')['input_ids'].shape
    print(output_length)
    max_length = max(max_length, length[1])
    max_output_length = max(max_output_length, output_length[1])
    print("=========")
    break
print("Max length: ", max_length)
print("Max output length: ", max_output_length)

# %%
count = 0
for batch in data_module.val_dataloader():
    print(batch['data']['graph_data'].edge_index[:, :20])
    count += 1
    if count == 10:
        break



# %%
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.model.GraphLlama import GraphLlamaForCausalLM
import torch

class args:
    model_key = "meta-llama/Llama-3.2-1B"

model_key = args.model_key.replace("/", "-").replace("..", "")
if "gpt" in args.model_key or "Llama" in model_key \
    or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
    hf_key = args.model_key.replace("_", "-")
    tokenizer = AutoTokenizer.from_pretrained(hf_key)
    tokenizer.padding_side = 'right'
    model = GraphLlamaForCausalLM.from_pretrained(hf_key)
    model_type = "decoder"
    append_eos = True

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
from src.model.gnn_models.config import load_cfg
import numpy as np
algorithm = 'bfs'
cfg = load_cfg("./src/model/gnn_models/configs/SAGE.yml")
specs = np.load(f".//src/model/specs/{algorithm}_specs.npy", allow_pickle=True).item()


model.config.graph_hidden_size = model.config.hidden_size
model_graph_dict = model.get_model().initialize_graph_modules(
    graph_tower="SAGE",
    specs=specs, cfg=cfg
)
tower = model.get_graph_tower() # .to(dtype=torch.float16, device=training_args.device)
# tower(batch['graph_data'])
# # %%
# model.model.graph_projector = torch.nn.Linear(cfg.MODEL.HIDDEN_DIM, model.config.hidden_size)

# %%
model.requires_grad_(False)
# only the graph tower is trainable
for p in model.get_model().get_graph_tower().parameters():
    p.requires_grad = True
for p in model.get_model().graph_projector.parameters():
    p.requires_grad = True

# %% 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch = batch['data']
batch = {k: v.to(device) for k, v in batch.items()}
model = model.to(device)

# %%
output = model(**batch)

# %%
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data

def to_torch(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    elif isinstance(value, torch.Tensor):
        return value
    else:
        return torch.tensor(value)
    
def convert_strA_to_adj(str_A):
    A = np.array([list(map(float, row.strip(' []').split())) for row in str_A.split(',') if row])
    return A

def to_data(input_str):
    data_dict = {}
    input_attributes = []
    
    # first get the edge index; create a fully connected graph 
    A_start = input_str.index('A: ')
    A_end = input_str.index(', initial_trace: ')
    str_A = input_str[A_start + 3: A_end] 
    A = convert_strA_to_adj(str_A)
    graph = nx.from_numpy_array(A, create_using=nx.DiGraph())
    edge_list = np.array(list(graph.edges())).T
    data_dict['edge_index'] = torch.tensor(edge_list, dtype=torch.long) # np.concatenate([edge_list, edge_list[::-1]], axis=1)
    num_nodes = A.shape[0]

    # add self loops
    unique_values = np.unique(A)
    is_weighted = unique_values.size != 2 or not np.all(unique_values == np.array([0,1]))
    if is_weighted:
        data_dict["weights"] = (A + np.eye(A.shape[0]))[data_dict["edge_index"][0], data_dict["edge_index"][1]]

    # Parse inputs
    additional_attributes = ['\ns:', ]
    for attribute in additional_attributes:
        if attribute in input_str:
            start = input_str.index(attribute)
            end = input_str.index('A:')
            value = int(input_str[start + len(attribute):end].strip(' ,'))
            feature = torch.zeros(num_nodes); feature[value] = 1
            input_attributes.append(attribute.strip('\n:'))
            data_dict[attribute.strip('\n:')] = feature
    
    data_dict['pos'] = torch.arange(0, 1, 1/num_nodes, dtype=torch.float)
    input_attributes.append('pos')
    
    data_dict = {k: to_torch(v) for k,v in data_dict.items()}
    data = Data(**data_dict)    
    data.inputs = input_attributes
    return data


# CLRSData(edge_index=[2, 76], pos=[16], length=7, s=[16], pi=[76], pi_h=[76, 7], reach_h=[16, 7], hints=[2], inputs=[2], outputs=[1])
# Data(edge_index=[2, 48], pos=[15], s=[15], inputs=[2])

# %%
data = to_data(test_sample['input'])

# %%
import numpy as np
for algorithm in ['bfs', "dfs", "topological_sort", "articulation_points", "bridges", 
                   "strongly_connected_components", "mst_kruskal", "mst_prim", 
                   "dijkstra", "bellman_ford", 'dag_shortest_paths', "floyd_warshall" ]:
    specs = np.load(f"./src/model/specs/{algorithm}_specs.npy", allow_pickle=True).item()
    specs = {k: v for k, v in specs.items() if v[0]=="input"}
    print(specs)



# Size 15
# bfs train length 514 output length 219 
    # 5 length 3797
    # 10 length 6894
    # 20 length 13181
    # 40 length 25631
# bellman_ford length 690 output length 250
    # 5 length 4643
    # 10 length 1
    # 20 length 16224
    # 40 length 31672
# dfs train length 508 output length 1366
    # 5 length 9868
    # 10 length 19228
    # 20 length 37948
# dijkstra length 688 output length 467
    # 5 length 6237
    # 10 length 11735
    # 20 length 22782
# mst_prim length 689 output length 467
    # 5 length 6233
    # 10 length 11744
    # 20 length 22753
# topological_sort length 516 output length 1259
    # 5 length 9143
    # 10 length 17464
    # 20 length 34208

# Size 4
# "floyd_warshall"
# 10 Max length:  1080
# 20 Max length:  1893
# 40 Max length:  3672
# 80 Max length:  7260
# 100 Max length: 9013
# 200 Max length: 17781
# 400 Max length: 35262

# bfs 846
# dfs 1617
# topological_sort 1965
# articulation_points 1731
# mst_kruskal 1760
# mst_prim 1098
# dijkstra: 1073
# bellman_ford 1029
# dag_shortest_paths 1469
# floyd_warshall 2183

# bfs 29
# dfs 101
# topological_sort 133
# articulation_points 128
# mst_kruskal 74
# mst_prim 38
# dijkstra: 38
# bellman_ford 29
# dag_shortest_paths 110
# floyd warshall: 110

# strongly_connected_components 2752
# strongly_connected_components 209

# bridges 5266 
# bridges: 506

# dijkstra: 37285
# topological sort: 72813
# articulation points: 65100
# bridges: 198322
# mst kruskal: 61370
# mst prim: 40640
# dag: 53790
# floyd warshall: 83690

# %%
count = 0
for batch in data_module.train_dataloader():
    # print(batch)
    print(tokenizer.batch_decode(batch['data']['input_ids'], skip_special_tokens=True)[0])
    count += 1
    if count == 200:
        break

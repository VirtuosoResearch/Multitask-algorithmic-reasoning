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
from transformers import AutoTokenizer
from src.custom.clrs_task_data_module import CLRSDataModule

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.mask_token_id = 128002 # special token
task_name = "bfs"

data_module = TextCLRSDataModule(
    task_names=[task_name],
    tokenizer=tokenizer,
    batch_size=8,
    inference_batch_size=8,
    max_input_length=128,
    max_output_length=128,
    train_lengths=[15],
    test_lengths=[15],
    eval_split=0.3,
    eval_all=True,
    use_few_shot=False,
    few_shot_k=0)
data_module.setup(stage="fit")

# %%
max_length = 0; max_output_length = 0
for test_sample in data_module.task_to_test_datasets[task_name]:
    print(test_sample['input'])
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

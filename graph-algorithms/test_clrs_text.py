# %% 
from clrs_text_tasks.graph_text_encoder import encode_graph
from clrs_text_tasks.utils import CLRSData, CLRSDataset, CLRSCollater

train_dataset = CLRSDataset(
     root="./data/CLRS",
     algorithm='strongly_connected_components', split="train", num_samples=1000, use_complete_graph=False)
# %%
from clrs_text_tasks.graph_text_encoder import encode_graph
import networkx as nx
import torch

data = train_dataset[0]
# %%
graph = nx.from_edgelist(data.edge_index.T.numpy())
edge_weights = {tuple(edge): weight for edge, weight in zip(data.edge_index.T.numpy(), data.weights.numpy())}
nx.set_edge_attributes(graph, values = edge_weights, name = 'weight')

# %%
graph_encoding = encode_graph(graph, "incident", add_weights=True)
print(graph_encoding)

# %%

def get_answer(data):
    answer = list(data.edge_index.T[data.pi==1])
    answer = sorted(answer, key=lambda x: x[0])
    answer = [str(x[1].item()) for x in answer]
    answer = ", ".join(answer)
    return answer

def get_intermediate_results(data):
    hints = data.pi_h
    intermediate_results = []
    for i in range(hints.shape[1]):
        answer = list(data.edge_index.T[data.pi_h[:, i]==1])
        if len(answer) < data.num_nodes: break
        answer = sorted(answer, key=lambda x: x[0])
        answer = [str(x[1].item()) for x in answer]
        answer = ", ".join(answer)
        intermediate_results.append(answer)
    return intermediate_results # exclude the last step since it is the answer

print(get_answer(data))
print(get_intermediate_results(data))
# %%
# one parameter source
problem_description = "In a single-source shortest-paths problem, we are given a weighted directed graph and a source node. We aim to find the shortest path starting from source node to every other node. \
For each node, each node has a pointer to its predecessor along the optimal path from the source. By convention, the source node points to itself. "

algorithm_description = "The Bellman-Ford algorithm starts by initializing the distance to the source vertex as 0 and all other vertices as infinity. \
It then iteratively relaxes each edge in the graph V-1 times where V is the number of vertices. \
Relaxing an edge involves checking whether the distance to the destination vertex can be reduced by taking the edge and, if so, updating the distance and the predecessor. \
After V-1 iterations, the algorithm performs an additional pass to check for negative-weight cycles. \
If any edge can still be relaxed, it indicates the presence of a negative-weight cycle, which is reported. Otherwise, the shortest paths from the source to all vertices are finalized."

question_prompt = "Question: Return the the predecessor nodes of all nodes in the shortest path to node {source} in alphabetical order."

answer_prompt = "Answer: {answer}"
answer = get_answer(data)

chain_of_thought_promt = "Steps: \n{steps}"
intermediate_results = get_intermediate_results(data)
# intermediate_results = [step for step in intermediate_results if step != answer ]

# %%
from clrs_text_tasks.graph_text_encoder import encode_graph
from clrs_text_tasks.utils import CLRSData, CLRSDataset, CLRSCollater
from clrs_text_tasks.tasks import BellmanFordTask
import json
import os

train_dataset = CLRSDataset(
     root="./data/CLRS",
     algorithm='bellman_ford', split="train", num_samples=1000, use_complete_graph=False)

task = BellmanFordTask()
add_cot = False

examples = task.prepare_examples(
        train_dataset, encoding_method="incident", add_description=True, add_cot=add_cot
    )

def write_examples(examples, output_path):
    with open(output_path + ".json", 'w') as file:
        for example in examples:
            json.dump(example, file)

root_dir = "./data/clrs_text_tasks"
file_name = task.name + '_' + 'train' + ('_cot' if add_cot else '')

write_examples(
    examples,
    os.path.join(root_dir, file_name),
)

# %%
val_dataset = CLRSDataset(
     root="./data/CLRS",
     algorithm='bellman_ford', split="val", num_samples=32, use_complete_graph=False)

examples = task.prepare_examples(
        val_dataset, encoding_method="incident", add_description=True, add_cot=add_cot
    )

file_name = task.name + '_' + 'val' + ('_cot' if add_cot else '')

write_examples(
    examples,
    os.path.join(root_dir, file_name),
)

# %%

for num_samples in range(2, 11, 2):
    examples = task.prepare_few_shot_examples(
            train_dataset, encoding_method="incident", num_samples=num_samples, add_cot=add_cot
        )
    file_name = task.name + '_' + 'train' + ('_cot' if add_cot else '') + f'_few_shot_{num_samples}'

    write_examples(
        examples,
        os.path.join(root_dir, file_name),
    )

# %%
from transformers import AutoTokenizer
from src.custom.clrs_task_data_module import CLRSDataModule

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.mask_token_id = 128002 # special token

data_module = CLRSDataModule(task_names=["bellman_ford"],
                prompt_styles=["zero_shot"],
                tokenizer=tokenizer,
                batch_size=8,
                inference_batch_size=8,
                max_input_length=5000,
                max_output_length=1024,
                eval_all=True)
data_module.setup(stage="fit")
# 3500 5000 7000 9000 12000
# cot 5000 7000 9000 11000 13000

# %%
for batch in data_module.train_dataloader():
    print(batch)
    break

tokenizer.batch_decode(batch['data']['input_ids'], skip_special_tokens=True)

# %%
print(tokenizer.batch_decode(batch['data']['input_ids'], skip_special_tokens=True)[0])

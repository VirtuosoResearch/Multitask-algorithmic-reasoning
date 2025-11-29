# %%
{"index": 0, "source_data": {
    "index": 0, 
    "input_prompt": "Determine whether or not there is a cycle in an undirected graph. In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. Given a graph, you need to output Yes or No, indicating whether there is a cycle in the graph. Q: The nodes are numbered from 0 to 4, and the edges are: (0, 3) (0, 4) (0, 1) (0, 2) (1, 4) (1, 2) (1, 3) (2, 3) (2, 4) (3, 4). Is there a cycle in this graph?", "answer": "### Yes", 
    "node_range": [2, 20], "edge_range": [0, 190]}, 
    "input_str": "Below is an instruction that describes a task. Write a response that appropriately completes the request step by step.\n\n### Instruction:\nDetermine whether or not there is a cycle in an undirected graph. In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. Given a graph, you need to output Yes or No, indicating whether there is a cycle in the graph. Q: The nodes are numbered from 0 to 4, and the edges are: (0, 3) (0, 4) (0, 1) (0, 2) (1, 4) (1, 2) (1, 3) (2, 3) (2, 4) (3, 4). Is there a cycle in this graph?\n\n### Response:", 
    "output_str": " This graph is a complete graph, meaning that every node is connected to every other node. Therefore, there are many cycles in this graph. For example, starting from node 0, we can go to node 1 (via edge 0-1), then to node 2 (via edge 1-2), then to node 3 (via edge 2-3), and back to node 0 (via edge 3-0). This forms a cycle [0-1-2-3-0] without revisiting any edge. Similarly, we can find cycles involving any other nodes. Thus, there is a cycle in this graph. ### Yes.", "task": "cycle"}


# %%
from datasets import load_dataset

tasks = ["connectivity", "bipartite", "cycle", "flow", "hamilton", "shortest", "substructure", "topology", "triangle"]

for task_name in ["hamilton"]:
    train_dataset = load_dataset("GraphWiz/GraphInstruct")['train']
    train_dataset = train_dataset.filter(lambda x: x['task'] == task_name)
    test_dataset = load_dataset("GraphWiz/GraphInstruct-Test", task_name)['test']
    print(len(train_dataset), len(test_dataset))

# %%
from src.custom.graphwiz_task_data_module import GraphWizDataModule
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("GraphWiz/LLaMA2-7B-DPO", trust_remote_code=True)
data_module = GraphWizDataModule(
    task_names=['triangle'], 
    tokenizer=tokenizer,
    batch_size=8,
    inference_batch_size=32,
    max_input_length=2048,
    max_output_length=1024,
    shuffle_train=True,
    eval_all=False,
    eval_split=0,
    downsample_ratio=1.0, # ratio of downsampling
    minimum_samples=100,
    minimum_samples_validation=100,
    downsample_seed=0,)
data_module.setup(stage="fit")

# %%
# measure input and output lengths
max_input_length = 0 
max_output_length = 0
for i in range(len(data_module.multitask_test_dataset.datasets[0])):
    sample = data_module.multitask_test_dataset.datasets[0][i]
    new_input_length = len(tokenizer(sample['input'])['input_ids'])
    if new_input_length > 2048:
        print(new_input_length)
    new_output_length = len(tokenizer(sample['output'])['input_ids'])
    if new_output_length > 1024:
        print(new_output_length)
    max_input_length = max(max_input_length, len(tokenizer(sample['input'])['input_ids']))
    max_output_length =  max(max_output_length, len(tokenizer(sample['output'])['input_ids']))
print(max_input_length, max_output_length)

# 1264 1753
# 2455 1045

# %%
for batch in data_module.train_dataloader():
    print(batch)
    break

tokenizer.decode(batch['data']['input_ids'][0])

# %%
import matplotlib.pyplot as plt

plt.hist(output_lengths, bins=100)
plt.vlines([128], 0, 800, color='r', linestyle='dashed', linewidth=1)


# %%
# %%
import graph_task as graph_task
import graph_task_utils as utils


GRAPHS_DIR = "../data/graphs"
TASK_DIR="../data/tasks"
TASK = "edge_existence"
algorithms = ['er']
text_encoders = [
    'adjacency',
    'incident',
    'coauthorship',
    'friendship',
    'south_park',
    'got',
    'social_network',
    'politician',
    'expert',
]


TASK_CLASS = {
    'edge_existence': graph_task.EdgeExistence,
    'node_degree': graph_task.NodeDegree,
    'node_count': graph_task.NodeCount,
    'edge_count': graph_task.EdgeCount,
    'connected_nodes': graph_task.ConnectedNodes,
    'cycle_check': graph_task.CycleCheck,
    'disconnected_nodes': graph_task.DisconnectedNodes,
    'reachability': graph_task.Reachability,
    'shortest_path': graph_task.ShortestPath,
    'maximum_flow': graph_task.MaximumFlow,
    'triangle_counting': graph_task.TriangleCounting,
    'node_classification': graph_task.NodeClassification,
}

# Loading the graphs.
graphs = []
generator_algorithms = []
for algorithm in algorithms:
    loaded_graphs = utils.load_graphs(
        GRAPHS_DIR,
        algorithm,
        'test',
    )
    graphs += loaded_graphs
    generator_algorithms += [algorithm] * len(loaded_graphs)

# Defining a task on the graphs
task = TASK_CLASS[TASK]()

# %%

# change this to not using tensorflow 
def create_example_feature(
    key,
    question,
    answer,
    algorithm,
    encoding_method,
    nnodes,
    nedges,
):
  """Create a tensorflow example from a datapoint."""
  key_feature = str(key)
  question_feature = question
  answer_feature = answer
  algorithm_feature = algorithm
  encoding_method_feature = encoding_method
  nnodes_feature = value=[nnodes]
  nedges_feature = value=[nedges]
  example_feats = {'id': key_feature,
          'question': question_feature,
          'answer': answer_feature,
          'algorithm': algorithm_feature,
          'text_encoding': encoding_method_feature,
          'nnodes': nnodes_feature,
          'nedges': nedges_feature,
      }
  return example_feats


def prepare_examples(
    examples_dict,
    encoding_method,
):
  """Create a list of tf.train.Example from a dict of examples."""
  examples = []
  for key, value in examples_dict.items():
    (
        question,
        answer,
        nnodes,
        nedges,
        algorithm,
    ) = (
        value['question'],
        value['answer'],
        value['nnodes'],
        value['nedges'],
        value['algorithm'],
    )
    examples.append(
        create_example_feature(
            key,
            question,
            answer,
            algorithm,
            encoding_method,
            nnodes,
            nedges,
        )
    )
  return examples


def create_zero_shot_task(
    task,
    graphs,
    generator_algorithms,
    text_encoders,
    cot = False,
):
  """Create a recordio file with zero-shot examples for the task."""
  examples = []
  for encoding_method in text_encoders:
    examples_dict = task.prepare_examples_dict(
        graphs, generator_algorithms, encoding_method
    )
    if cot:
      for key in examples_dict.keys():
        examples_dict[key]['question'] += "Let's think step by step. "
    examples += prepare_examples(examples_dict, encoding_method) # TODO: change this
  return examples

split='test'
cot = False
zero_shot_examples = create_zero_shot_task(
    task, graphs, generator_algorithms, text_encoders, cot=False
)

# %%
import json
file_name = task.name + ('_zero_cot_' if cot else '_zero_shot_')
file_name += split

def write_examples(examples, output_path):
    with open(output_path + ".json", 'w') as file:
        for example in examples:
            json.dump(example, file)

if not os.path.exists(TASK_DIR):
    os.makedirs(TASK_DIR)

write_examples(
    zero_shot_examples,
    os.path.join(TASK_DIR, file_name),
)

# %%
from datasets import load_dataset

# dataset = load_dataset("json", data_files="./data/tasks/nodes_20_21/node_degree_zero_shot_er_train.json")['train']

# GSM8K
# dataset = load_dataset("openai/gsm8k", "main")

# MATH 
# ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
dataset = load_dataset("EleutherAI/hendrycks_math", 'algebra')

# %%
from src.custom.gsm8k_data_module import GSM8KDataModule
from src.custom.math_data_module import MATHDataModule
from src.custom.clrs_text_task_data_module import TextCLRSDataModule
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

# data_module = TextCLRSDataModule(
#     tokenizer=tokenizer,
#     batch_size=8,
#     inference_batch_size=32,
#     max_input_length=512,
#     max_output_length=512,
#     shuffle_train=True,
#     eval_split=0,
#     downsample_ratio=1.0,  # ratio of downsampling
#     minimum_samples=10000,
#     minimum_samples_validation=10000,
#     downsample_seed=0,
#     only_answer_output=True
# )



# "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\

data_module =  TextCLRSDataModule(
    task_names=["floyd_warshall"],
    tokenizer=tokenizer,
    batch_size=4,
    inference_batch_size=4,
    max_input_length=512,
    max_output_length=512,
    eval_all=True,
    eval_split=0.2,
    downsample_ratio=0.01,
    minimum_samples=1000,
    minimum_samples_validation=1000,
    downsample_seed=0,
    only_answer_output=False,
    train_lengths=[5],
    test_lengths=[5],
    use_few_shot=False, 
    few_shot_k=0,)

data_module.setup(stage="fit")

for batch in data_module.train_dataloader():
    break

max_input_length = 0
max_output_length = 0
output_lengths = []
for i in range(len(data_module.multitask_train_dataset.datasets[0])):
    sample = data_module.multitask_train_dataset.datasets[0][i]
    max_input_length = max(max_input_length, len(tokenizer(sample['input'])['input_ids']))
    max_output_length =  max(max_output_length, len(tokenizer(sample['output'])['input_ids']))
    output_lengths.append(len(tokenizer(sample['output'])['input_ids']))
print(max_input_length, max_output_length)

# %%
from transformers import AutoTokenizer
from src.custom.algorithm_task_data_module import AlgorithmDataModule

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer.pad_token_id = tokenizer.eos_token_id

lens = []
for i in range(len(dataset)):
    lens.append(len(tokenizer(dataset[i]['question'])['input_ids']) + len(tokenizer(dataset[i]['answer'])['input_ids']))
    print(lens[-1])
print(max(lens), min(lens), sum(lens) / len(lens))

# %%
import re
import numpy as np
from transformers import AutoTokenizer
def extract_target_node(text):
    """
    Extract the target node ID from a question like 'Q: What is the degree of node 12?'
    Returns an integer node ID if found, otherwise raises ValueError.
    """
    match = re.search(r'What\sis\sthe\sdegree\sof\snode\s+(\d+)', text)
    if match:
        return int(match.group(1))
    else:
        return -1

def extract_edges(text):
    # Extract edge tuples from the format "(i, j)"
    edge_tuples = re.findall(r'\((\d+),\s*(\d+)\)', text)
    edges = [(int(i), int(j)) for i, j in edge_tuples]
    return edges

def get_one_hop_subgraph_spans(text, target_node, include_prefix=True):
    edges = extract_edges(text)
    neighbors = {j if i == target_node else i for i, j in edges if target_node in (i, j)}
    relevant_nodes = neighbors.union({target_node})

    subgraph_spans = []

    # Include edge spans involving the target node
    for match in re.finditer(r'\((\d+),\s*(\d+)\)', text):
        i, j = int(match.group(1)), int(match.group(2))
        if i == target_node or j == target_node:
            subgraph_spans.append((match.start(), match.end()))

    # Include individual node mentions
    for node in relevant_nodes:
        for match in re.finditer(rf',\s\b{node}\b,', text):
            subgraph_spans.append((match.start()+2, match.end()-1))

    # Include the introductory explanation sentence
    if include_prefix:
        prefix = "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. G describes a graph among nodes"
        prefix_start = text.find(prefix)
        if prefix_start != -1:
            subgraph_spans.append((prefix_start, prefix_start + len(prefix)))

    return subgraph_spans

def create_token_mask_node_degree(text, tokenizer, target_node, max_length=512):
    subgraph_spans = get_one_hop_subgraph_spans(text, target_node)

    # Tokenize with offset mapping to map tokens to character spans
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, max_length=max_length, truncation=True, padding='max_length')
    offsets = encoding["offset_mapping"]
    token_mask = []

    for start, end in offsets:
        # Check if this token span overlaps any subgraph span
        overlaps = any(not (end <= s or start >= e) for s, e in subgraph_spans)
        token_mask.append(1 if overlaps else 0)

    return np.array(token_mask, dtype=np.int64)

# Example usage:
text = '''In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. G describes a graph among nodes 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, and 14.
The edges in G are: (0, 1) (0, 6) (1, 2) (1, 3) (1, 7) (1, 10) (1, 11) (1, 12) (1, 13) (1, 14) (2, 4) (2, 5) (2, 6) (2, 8) (2, 10) (2, 11) (2, 12) (2, 13) (2, 14) (3, 5) (3, 8) (3, 9) (3, 13) (4, 6) (4, 8) (4, 9) (4, 14) (5, 7) (5, 8) (5, 11) (5, 12) (6, 7) (6, 8) (6, 10) (7, 9) (7, 10) (7, 12) (7, 13) (8, 10) (9, 11) (9, 12) (9, 14) (10, 11) (10, 13) (11, 13) (11, 14) (12, 14).
Q: What is the degree of node 12?
A: '''
target_node = extract_target_node(text)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
mask = create_token_mask_node_degree(text, tokenizer, target_node, max_length=512)

print([tokens[i] for i in range(len(tokens)) if mask[i] == 1])

# %%
data_module = AlgorithmDataModule(
    task_names=["node_degree"],
    prompt_styles=["zero_shot"],
    text_encoders=["adjacency"],
    node_range=(15, 16),
    tokenizer=tokenizer,

)
data_module.setup()
# %%
for train_batch in data_module.train_dataloader():
    print(train_batch['input_ids'].shape)
    print(train_batch['attention_mask'].shape)
    print(train_batch['labels'].shape)
    print(train_batch['text_encoding'])
    print(train_batch['nnodes'])
    print(train_batch['nedges'])
    break
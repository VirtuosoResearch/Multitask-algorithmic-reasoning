# %%
from clrs_text_tasks.graph_text_encoder import encode_graph
from clrs_text_tasks.utils import CLRSData, CLRSDataset, CLRSCollater
from clrs_text_tasks.tasks import BellmanFordTask
from clrs_text_tasks import task_name_to_tasks
import json
import os

# %%

task_name = "dfs"
add_cot = True

train_dataset = CLRSDataset(
     root="./data/CLRS",
     algorithm=task_name, split="train", num_samples=1000, use_complete_graph=False)

task = task_name_to_tasks[task_name]()

examples = task.prepare_examples(
        train_dataset, encoding_method="incident", add_description=True, add_cot=add_cot
    )

def write_examples(examples, output_path):
    with open(output_path + ".json", 'w') as file:
        for example in examples:
            json.dump(example, file)

root_dir = "./data/clrs_text_tasks"
file_name = task.name + '_' + 'train' + ('_cot' if add_cot else '') + '_zero_shot'

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
val_dataset = CLRSDataset(
     root="./data/CLRS",
     algorithm=task_name, split="val", num_samples=32, use_complete_graph=False)

examples = task.prepare_examples(
        val_dataset, encoding_method="incident", add_description=True, add_cot=add_cot
    )

file_name = task.name + '_' + 'val' + ('_cot' if add_cot else '') + '_zero_shot'

write_examples(
    examples,
    os.path.join(root_dir, file_name),
)

# %%
val_dataset = CLRSDataset(
     root="./data/CLRS",
     algorithm=task_name, split="test", num_samples=32, use_complete_graph=False)

examples = task.prepare_examples(
        val_dataset, encoding_method="incident", add_description=True, add_cot=add_cot
    )

file_name = task.name + '_' + 'test' + ('_cot' if add_cot else '') + '_zero_shot'

write_examples(
    examples,
    os.path.join(root_dir, file_name),
)

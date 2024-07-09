# # %%
# %pwd
# %cd ../

# %%
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from data_loader.utils import format_algorithm_example, generate_simple_algorithm_example
import datasets
from datasets import Dataset

class args:
    model_name = "gpt2"
    incontext_k = 0

train_datasets = []
algorithms = ["bfs"]
csv_dirs = ["bfs_data_10"]

for algorithm, csv_dir in zip(algorithms, csv_dirs):
    # load data
    file_name = f"./data/{algorithm}/{csv_dir}.csv"
    instance_df = pd.read_csv(file_name, index_col=0)
    num_of_instances = instance_df.shape[0]
    train_data = []

    def gen():
        for i in range(num_of_instances):
            yield generate_simple_algorithm_example(instance_df, i, k=args.incontext_k)
            # yield format_algorithm_example(instance_df, i, include_input=True, include_inter_results=True, include_answer=True)

    train_dataset = Dataset.from_generator(generator=gen)
    train_datasets.append(train_dataset)

dataset = datasets.concatenate_datasets(train_datasets)

# %%
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["input"]

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(vocab_size=250, special_tokens=[""])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

from transformers import GPT2TokenizerFast

wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
wrapped_tokenizer.save_pretrained("./tokenizers/gpt2_bfs")

# %%
tokenizer = AutoTokenizer.from_pretrained("./tokenizers/gpt2_bfs", use_fast=True, padding_side="left")

# %%
train_dataset = train_dataset.train_test_split(test_size=0.1)
train_dataset, valid_dataset = train_dataset["train"], train_dataset["test"]

# %%
from data_loader.collators import CLMCollator
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                            # torch_dtype=torch.float16, 
                                            # ignore_mismatched_sizes=True, n_positions=
                                            )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = 0

data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, collate_fn=CLMCollator(tokenizer, max_length=16))

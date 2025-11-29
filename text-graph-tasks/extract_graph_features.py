# %%
import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def initialize_model(args):
    model_key = args.model_key.replace("/", "-").replace("..", "")
    if "gpt" in args.model_key or "Llama" in model_key \
        or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
        hf_key = args.model_key.replace("_", "-")
        tokenizer = AutoTokenizer.from_pretrained(hf_key)
        tokenizer.padding_side = 'right'
        if args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": args.devices[0]}, attn_implementation="flash_attention_2") 
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_key, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        model_type = "decoder"
        append_eos = True
    elif "flan" in model_key:
        hf_key = "google/{}".format(model_key.replace("_", "-"))
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False  # t5 tokenizers already append eos
    else:
        raise NotImplementedError(args.model_key)
    
    if args.train_lora:
        if args.model_key == "gpt2": # for gpt2, we generally use full model
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["c_attn", "c_proj", "c_fc"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif args.model_key == "EleutherAI/gpt-neox-20b":
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query_key_value"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif "flan" in args.model_key:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q", "k", "v"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        else:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return model, tokenizer, hf_key, model_type, append_eos

class args:
    model_key = "meta-llama/Llama-3.2-1B"
    use_qlora = True
    devices = [0]
    train_lora = True
    lora_rank = 4
    lora_alpha = 32

model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
import pytorch_lightning as pl
import torch
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, IterableDataset
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import *
from torch.utils.data import BatchSampler
from src.utils.multitask_dataset import MultitaskDataset, MultitaskBatchSampler, MultitaskCollator
from datasets import load_dataset

@dataclass
class CasualLMInstructionCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None # maximum length of the output
    max_target_length: Optional[int] = None # maximum length of the input
    pad_to_multiple_of: Optional[int] = None 
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
                return_tensors = self.return_tensors

        converted_batch = batch

        # prepare input sources
        sources = []; source_lengths = []
        for instance in converted_batch:
            source = instance["input"]
            source = source.replace("\n", " ")
            source = " ".join(source.split())
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
            source_lengths.append(min(len(tokenized_source), self.max_source_length))

        labels = []; label_lengths = []
        for instance in converted_batch:
            label = instance["output"]
            label = label.replace("\n", " ")
            label = " ".join(label.split())
            tokenized_label = self.tokenizer(label)["input_ids"]
            if len(tokenized_label) <= self.max_target_length:
                labels.append(label)
            else:
                labels.append(self.tokenizer.decode(tokenized_label[:self.max_target_length], skip_special_tokens=True))
            label_lengths.append(min(len(tokenized_label), self.max_target_length))

        inputs = [source + " " + label for source, label in zip(sources, labels)]

        model_inputs = self.tokenizer(
                text = inputs, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True)
        
        # prepare labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        label_mask = model_inputs["attention_mask"].clone().bool()
        model_inputs["labels"] = model_inputs["labels"].masked_fill(~label_mask, self.label_pad_token_id)
        for i, length in enumerate(source_lengths):
            model_inputs["labels"][i, :length] = self.label_pad_token_id            

        if "weights" in converted_batch[0]:
            model_inputs["weights"] = torch.Tensor([instance["weights"] for instance in converted_batch])

        if "residuals" in converted_batch[0]:
            model_inputs["residuals"] = torch.Tensor([instance["residuals"] for instance in converted_batch])
        
        return model_inputs

class convert_format:

    def __call__(self, examples):
        examples["input"] = examples["question"][:]
        examples["output"] = examples["answer"][:]
        return examples
    
class convert_format_to_extract_features:
    # we use the <|reserved_special_token_0|> as the mask token for every node positions
    def __call__(self, examples):
        examples["input"] = examples["question"][:]

        # get output positions 
        num_nodes = []; examples["output"] = []
        for answer in examples["answer"]:
            num_nodes.append(len(answer.split(", ")))
            examples["output"].append("".join(["<|reserved_special_token_0|>"]*num_nodes[-1]))
        return examples

class CLRSDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        task_names, 
        prompt_styles,
        tokenizer,
        batch_size=8,
        inference_batch_size=32,
        max_input_length=512,
        max_output_length=64,
        shuffle_train=True,
        eval_all=False,
        downsample_ratio=1.0, # ratio of downsampling
        minimum_samples=100,
        minimum_samples_validation=100,
        downsample_seed=0,
        extract_features=False,
    ):
        super().__init__()

        self.task_names = task_names # task_name
        self.prompt_styles = prompt_styles # zero_shot, zero_cot, few_shot, few_cot
        # "adjacency" "incident" "friendship" "south_park" "got" "politician"
        # "social_network" "expert" "coauthorship" "random" 

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.batch_size = batch_size
        if inference_batch_size is None:
            self.inference_batch_size = batch_size
        else:
            self.inference_batch_size = inference_batch_size
        self.shuffle_train = shuffle_train
        self.eval_all = eval_all

        self.downsample_rate = downsample_ratio
        self.downsample_seed = downsample_seed
        self.minimum_sample = minimum_samples
        self.minimum_sample_validation = minimum_samples_validation
        self.extract_features = extract_features

    def setup(self, stage=None):
        self.task_to_train_datasets = {}
        self.task_to_valid_datasets = {}
        self.task_to_test_datasets = {}
        self.task_to_collators = {}
        self.task_to_templates = {}
        for i, task_name in enumerate(self.task_names):
            prompt_style = self.prompt_styles[i]

            # Split the dataset into train and validation
            task_file_dir = "./data/clrs_text_tasks/{}_train_{}.json".format(task_name, prompt_style)
            train_dataset = load_dataset("json", data_files=task_file_dir)['train']
            # fileter out the examples by the text encoder
            column_names = train_dataset.column_names
            # convert the input and output format
            if self.extract_features:
                column_names.remove("answer")
                train_dataset = train_dataset.map(convert_format_to_extract_features(), batched=True, remove_columns=column_names)
            else:
                train_dataset = train_dataset.map(convert_format(), batched=True, remove_columns=column_names)

            task_file_dir = "./data/clrs_text_tasks/{}_val_zero_shot.json".format(task_name)
            eval_dataset = load_dataset("json", data_files=task_file_dir)['train']
            # fileter out the examples by the text encoder
            column_names = eval_dataset.column_names
            # convert the input and output format
            if self.extract_features:
                column_names.remove("answer")
                eval_dataset = eval_dataset.map(convert_format_to_extract_features(), batched=True, remove_columns=column_names)
            else:
                eval_dataset = eval_dataset.map(convert_format(), batched=True, remove_columns=column_names)
            
            task_file_dir = "./data/clrs_text_tasks/{}_test_zero_shot.json".format(task_name)
            predict_dataset = load_dataset("json", data_files=task_file_dir)['train']
            # fileter out the examples by the text encoder
            column_names = predict_dataset.column_names
            # convert the input and output format
            if self.extract_features:
                column_names.remove("answer")
                predict_dataset = predict_dataset.map(convert_format_to_extract_features(), batched=True, remove_columns=column_names)
            else:
                predict_dataset = predict_dataset.map(convert_format(), batched=True, remove_columns=column_names)

            # Downsample the dataset if needed
            if self.downsample_rate < 1.0:
                rng = np.random.default_rng(self.downsample_seed)
                permutations = rng.permutation(len(train_dataset))
                min_sample = max(int(self.minimum_sample), int(self.downsample_rate*len(train_dataset)))
                train_dataset = train_dataset.select(permutations[:min_sample])

            if self.downsample_rate < 1.0:
                rng = np.random.default_rng(self.downsample_seed)
                permutations = rng.permutation(len(eval_dataset))
                min_sample = max(int(self.minimum_sample_validation), int(self.downsample_rate*len(eval_dataset)))
                eval_dataset = eval_dataset.select(permutations[:min_sample])

            if self.downsample_rate < 1.0:
                rng = np.random.default_rng(self.downsample_seed)
                permutations = rng.permutation(len(predict_dataset))
                min_sample = max(int(self.minimum_sample_validation), int(self.downsample_rate*len(predict_dataset)))
                predict_dataset = predict_dataset.select(permutations[:min_sample])
            
            extended_task_name = task_name + "_" + prompt_style
            print("Task: {} train dataset size: {} validation dataset size: {} test dataset size: {}".format(extended_task_name, len(train_dataset), len(eval_dataset), len(predict_dataset)))
            self.task_to_train_datasets[extended_task_name] = train_dataset
            self.task_to_valid_datasets[extended_task_name] = eval_dataset
            self.task_to_test_datasets[extended_task_name] = predict_dataset
            self.task_to_collators[extended_task_name] = CasualLMInstructionCollator(self.tokenizer, padding="max_length", 
                                                    max_source_length=self.max_input_length, max_target_length=self.max_output_length)

        self.multitask_train_dataset = MultitaskDataset(self.task_to_train_datasets)
        self.multitask_valid_dataset = MultitaskDataset(self.task_to_valid_datasets)
        self.multitask_test_dataset = MultitaskDataset(self.task_to_test_datasets)
        self.multitask_collator = MultitaskCollator(self.task_to_collators)
        self.multitask_train_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_train_datasets.values()])), 
                                                                batch_size=self.batch_size, drop_last=False, task_to_datasets=self.task_to_train_datasets, shuffle=self.shuffle_train)
            # self.task_to_train_datasets, self.batch_size, shuffle=True)
        self.multitask_valid_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_valid_datasets.values()])), 
                                                                batch_size=self.inference_batch_size, drop_last=False, task_to_datasets=self.task_to_valid_datasets, shuffle=False)
            # self.task_to_valid_datasets, self.inference_batch_size, shuffle=False)

        if hasattr(self, "residuals") and hasattr(self, "weights"):
            cur_len = 0
            for extended_task_name, train_dataset in self.task_to_train_datasets.items():
                self.task_to_train_datasets[extended_task_name] = train_dataset.add_column("weights", self.weights[cur_len: cur_len+len(train_dataset)]) # add weights to train dataset
                cur_len += len(train_dataset)

            cur_len = 0
            for extended_task_name, train_dataset in self.task_to_train_datasets.items():
                self.task_to_train_datasets[extended_task_name] = train_dataset.add_column("residuals", self.residuals[cur_len: cur_len+len(train_dataset)])
                cur_len += len(train_dataset)

            print("Weights and residuals loaded!", "Weights mean: ", self.weights.mean(), "Residuals mean: ", self.residuals.mean())

    def train_dataloader(self):
        return DataLoader(
            self.multitask_train_dataset,
            batch_sampler=self.multitask_train_sampler,
            collate_fn=self.multitask_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.multitask_valid_dataset,
            batch_sampler=self.multitask_valid_sampler,
            collate_fn=self.multitask_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.multitask_test_dataset,
            batch_sampler=self.multitask_valid_sampler,
            collate_fn=self.multitask_collator,
        )

# %%

# tokenizer.mask_token_id = None # special token
task_name = "bfs"
data_module = CLRSDataModule(task_names=[task_name],
                prompt_styles=["zero_shot"],
                tokenizer=tokenizer,
                batch_size=4,
                inference_batch_size=4,
                max_input_length=2000,
                max_output_length=256,
                eval_all=True,
                extract_features=True,
                shuffle_train=False)
data_module.setup(stage="fit")

# 3500 5000 7000 9000 12000
# cot 5000 7000 9000 11000 13000

# %%
for batch in data_module.train_dataloader():
    print(batch)
    break

tokenizer.batch_decode(batch['data']['input_ids'], skip_special_tokens=True)
print(tokenizer.batch_decode(batch['data']['input_ids'], skip_special_tokens=True)[0])

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
if not args.use_qlora:
    model.to(device)

train_node_features = []
with torch.no_grad():
    for batch in data_module.train_dataloader():
        batch = {k: v.to(device) for k, v in batch['data'].items()}
        outputs = model(**batch, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1] # last layer
        hidden_states = hidden_states[:, :-1].contiguous()
        batch["labels"] = batch["labels"][:, 1:].contiguous()
        for batch_idx in range(len(batch["labels"])):
            batch_node_features = hidden_states[batch_idx][batch["labels"][batch_idx] == 128002] # 128002 is the mask token
            train_node_features.append(batch_node_features.detach().type(torch.float).to("cpu").numpy())

train_node_features = np.stack(train_node_features, axis=0)
np.save(f"{task_name}_train_node_features.npy", train_node_features)

# %%
val_node_features = []
with torch.no_grad():
    for batch in data_module.val_dataloader():
        batch = {k: v.to(device) for k, v in batch['data'].items()}
        outputs = model(**batch, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1] # last layer
        hidden_states = hidden_states[:, :-1].contiguous()
        batch["labels"] = batch["labels"][:, 1:].contiguous()
        for batch_idx in range(len(batch["labels"])):
            batch_node_features = hidden_states[batch_idx][batch["labels"][batch_idx] == 128002] # 128002 is the mask token
            val_node_features.append(batch_node_features.detach().type(torch.float).to("cpu").numpy())

val_node_features = np.stack(val_node_features, axis=0)
np.save(f"{task_name}_val_node_features.npy", val_node_features)
# %%

train_node_labels = []
for data in data_module.task_to_train_datasets[f"{task_name}_zero_shot"]:
    tmp_labels = data['answer'].split(", ")
    tmp_labels = np.array([int(label) for label in tmp_labels])
    train_node_labels.append(tmp_labels)

train_node_labels = np.stack(train_node_labels, axis=0)
np.save(f"{task_name}_train_node_labels.npy", train_node_labels)

# %%
val_node_labels = []
for data in data_module.task_to_valid_datasets[f"{task_name}_zero_shot"]:
    tmp_labels = data['answer'].split(", ")
    tmp_labels = np.array([int(label) for label in tmp_labels])
    val_node_labels.append(tmp_labels)

val_node_labels = np.stack(val_node_labels, axis=0)
np.save(f"{task_name}_val_node_labels.npy", val_node_labels)

# %%
import numpy as np
train_node_features = np.load(f"{task_name}_train_node_features.npy")
train_node_labels = np.load(f"{task_name}_train_node_labels.npy")

val_node_features = np.load(f"{task_name}_val_node_features.npy")
val_node_labels = np.load(f"{task_name}_val_node_labels.npy")

# %%
num_training = 1000
train_node_features = train_node_features[:num_training]
train_node_labels = train_node_labels[:num_training]

train_node_features = train_node_features.reshape( -1, train_node_features.shape[-1])
val_node_features = val_node_features.reshape( -1, val_node_features.shape[-1])
train_node_labels = train_node_labels.reshape(-1)
val_node_labels = val_node_labels.reshape(-1)

# train_node_features = train_node_features[:, 0, :]
# val_node_features = val_node_features[:, 0, :]
# train_node_labels = train_node_labels[:, 0]
# val_node_labels = val_node_labels[:, 0]

# normalize the features to uniform norm
# train_node_features = train_node_features / np.linalg.norm(train_node_features, axis=1, keepdims=True)
# val_node_features = val_node_features / np.linalg.norm(val_node_features, axis=1, keepdims=True)

# %%
# use a simple Linear layer for classification 
import torch
import torch.nn as nn

num_class = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Linear(train_node_features.shape[-1], 256),
    # nn.ReLU(),
    nn.Linear(256, num_class))
model.to(device)

criterion = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.Tensor(train_node_features), torch.Tensor(train_node_labels).long()), 
    batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.Tensor(val_node_features), torch.Tensor(val_node_labels).long()), 
    batch_size=256, shuffle=False)

# optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
num_epochs = 1000

# %%
for epoch in range(1, num_epochs+1):
    training_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        log_probs = nn.functional.log_softmax(output, dim=1)
        loss = torch.nn.functional.nll_loss(log_probs, target)
        # criterion(output, target)
        loss.backward()
        training_loss += loss.item()
        optimizer.step()
    
    if epoch % 10 == 0:
        print("Epoch: {}, Training Loss: {}".format(epoch, training_loss/len(train_loader)))

        correct = 0
        total = 0
        with torch.no_grad():
            test_loss = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                test_loss += nn.functional.nll_loss(log_probs, target).item()
                predicted = torch.argmax(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            

        print('Epoch: {}, Test Loss: {}, Accuracy: {}'.format(epoch, test_loss/len(test_loader), 100 * correct / total))

 #%%
'''
Bellman ford: 62.5
BFS: 81.2
'''
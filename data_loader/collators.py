import torch
import numpy as np

class CLMCollator:
    def __init__(self, tokenizer, max_length=1024, padding="max_length", label_pad_token_id=-100,
                 return_indices = False, output_columns = ["output"], **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.label_pad_token_id = label_pad_token_id
        self.return_indices = return_indices
        self.output_columns = output_columns

    def __call__(self, examples):
        input_lengths = []
        for instance in examples:
            input = instance["input"]
            tokenized_source = self.tokenizer(input)["input_ids"]
            input_lengths.append(len(tokenized_source))

        output_lengths = []
        for instance in examples:
            output = instance["output"]
            tokenized_source = self.tokenizer(output)["input_ids"]
            output_lengths.append(len(tokenized_source))

        task_indices = []
        for instance in examples:
            tmp_task_indices = {}; cur_index = 0
            for column in self.output_columns:
                length = len(instance[f"column_{column}"].split()) if f"column_{column}" in instance else 0
                tmp_task_indices[f"column_{column}"] = [i for i in range(cur_index, cur_index + length)]
                cur_index += length
            task_indices.append(tmp_task_indices)

        # Deprecated
        if "inter_results" in examples[0]: 
            inputs = [instance['input'] + instance['inter_results'] + instance['output'] for instance in examples]
        # Deprecated
        else:
            inputs = [instance['input'] + instance['output'] for instance in examples]

        model_inputs = self.tokenizer(
                text = inputs, 
                max_length=self.max_length, 
                padding=self.padding,
                return_tensors="pt",
                truncation=True)
        
        # prepare labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        label_mask = model_inputs["attention_mask"].bool()
        model_inputs["labels"] = model_inputs["labels"].masked_fill(~label_mask, self.label_pad_token_id)
        for i, length in enumerate(input_lengths):
            model_inputs["labels"][i, :length] = self.label_pad_token_id

        model_inputs["task_indices"] = task_indices

        # Deprecated
        if "step" in examples[0]:
            step_idxes = []
            for instance in examples:
                step_idxes.append(int(instance["step"]))
            model_inputs['steps'] = torch.LongTensor(step_idxes)
        
        if "index_0" in examples[0] and "index_1" in examples[0]:
            index_0 = []
            index_1 = []
            for instance in examples:
                index_0.append(int(instance["index_0"]))
                index_1.append(int(instance["index_1"]))
            indexes = np.concatenate([np.array(index_0).reshape(-1, 1), np.array(index_1).reshape(-1, 1)], axis=1)
            model_inputs['indexes'] = torch.LongTensor(indexes)
        # Deprecated

        return model_inputs
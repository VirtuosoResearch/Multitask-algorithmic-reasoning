import torch
import torch_geometric.nn as nn
from inspect import signature
from loguru import logger

from .encoder import Encoder
from .decoder import Decoder, grab_outputs, output_mask
from .processor import Processor
from ..utils import stack_hidden   

from abc import ABC
import torch_geometric.nn as pyg_nn
from core.models.processor import _get_processor

def stack_hints(hints):
    return {k: torch.stack([hint[k] for hint in hints], dim=-1) for k in hints[0]} if hints else {}

class MMOE_Processor(torch.nn.Module, ABC):
    def __init__(self, cfg, num_tasks, num_experts, has_randomness=False):
        super().__init__()
        self.cfg = cfg        
        processor_input = self.cfg.MODEL.HIDDEN_DIM*3 if self.cfg.MODEL.PROCESSOR_USE_LAST_HIDDEN else self.cfg.MODEL.HIDDEN_DIM*2
        if has_randomness:
            processor_input += 1
        
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        for i in range(num_experts):
            self.experts.append(_get_processor(self.cfg.MODEL.PROCESSOR.NAME)(in_channels=processor_input, out_channels=self.cfg.MODEL.HIDDEN_DIM, **self.cfg.MODEL.PROCESSOR.KWARGS[0]))    
        for i in range(num_tasks):
            self.gates.append(nn.Linear(processor_input, num_experts))             
        
        self.norms = torch.nn.ModuleList()
        if self.cfg.MODEL.PROCESSOR.LAYERNORM.ENABLE:
            for _ in range(num_experts):
                self.norms.append(pyg_nn.LayerNorm(self.cfg.MODEL.HIDDEN_DIM, mode=self.cfg.MODEL.PROCESSOR.LAYERNORM.MODE))
        self.core = self.experts[0]
        self._core_requires_last_hidden = "last_hidden" in signature(self.experts[0].forward).parameters

    def forward(self, input_hidden, hidden, last_hidden, batch_assignment, task_index, randomness=None, **kwargs):
        stacked = stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.PROCESSOR_USE_LAST_HIDDEN)
        if randomness is not None:
            stacked = torch.cat((stacked, randomness.unsqueeze(1)), dim=-1)
        if self._core_requires_last_hidden:
            kwargs["last_hidden"] = last_hidden
        
        expert_outs = []
        for i in range(self.num_experts):
            out = self.experts[i](stacked, **kwargs)
            if self.cfg.MODEL.PROCESSOR.LAYERNORM.ENABLE:
                # norm
                out = self.norms[i](out, batch=batch_assignment)
            expert_outs.append(out)
        expert_outs = torch.stack(expert_outs, dim=1) # [Batch, Experts, Dim]
        
        gate_outs = self.gates[task_index](stacked) # [Batch, Experts]
        gate_outs = torch.nn.functional.softmax(gate_outs, dim=-1)
        gate_outs = gate_outs.unsqueeze(-1) # [Batch, Experts, 1]
        output = torch.sum(expert_outs * gate_outs, dim=1)
            
        return output

    def has_edge_weight(self):
        return "edge_weight" in signature(self.experts[0].forward).parameters
    
    def has_edge_attr(self):
        return "edge_attr" in signature(self.experts[0].forward).parameters

class MMOE_EncodeProcessDecode(torch.nn.Module):
    def __init__(self, task_to_specs, cfg, num_experts):
        super().__init__()
        self.cfg = cfg
        self.tasks = list(task_to_specs.keys())
        self.task_to_specs = task_to_specs

        self.num_experts = num_experts
        self.num_tasks = len(self.tasks)

        self.has_randomness = False # 'randomness' in specs # No randomness for the model now
        self.processor = torch.nn.ModuleList()
        for _ in range(self.cfg.MODEL.MSG_PASSING_STEPS):
            self.processor.append(MMOE_Processor(cfg, self.num_tasks, self.num_experts, self.has_randomness))
        # Create task-specific encoders and decoders
        self.encoders = torch.nn.ModuleDict()
        self.decoders = torch.nn.ModuleDict()
        for task, specs in task_to_specs.items():
            self.encoders[task] = Encoder(specs, self.cfg.MODEL.HIDDEN_DIM)

        decoder_input = self.cfg.MODEL.HIDDEN_DIM*3 if self.cfg.MODEL.DECODER_USE_LAST_HIDDEN else self.cfg.MODEL.HIDDEN_DIM*2
        for task, specs in task_to_specs.items():
            self.decoders[task] = Decoder(specs, decoder_input, no_hint=self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT == 0.0)

        if not self.processor[0].has_edge_weight() and not self.processor[0].has_edge_attr():
            if "A" in specs:
                logger.warning(f"Processor {self.cfg.MODEL.PROCESSOR.NAME} does neither support edge_weight nor edge_attr, but the algorithm requires edge weights.")
                raise ValueError(f"Processor {self.cfg.MODEL.PROCESSOR.NAME} does neither support edge_weight nor edge_attr, but the algorithm requires edge weights.")
        elif self.processor[0].has_edge_weight():
            self.edge_weight_name = "edge_weight"
        elif self.processor[0].has_edge_attr():
            self.edge_weight_name = "edge_attr"

        if self.cfg.MODEL.GRU.ENABLE:
            self.gru = torch.nn.GRUCell(self.cfg.MODEL.HIDDEN_DIM, self.cfg.MODEL.HIDDEN_DIM)
        
    def process_weights(self, batch):
        if self.edge_weight_name == "edge_attr":
            if batch.weights.dim() == 1:
                return batch.weights.unsqueeze(-1).type(torch.float32)
            return batch.weights.type(torch.float32)
        else:
            return batch.weights
        
    def forward(self, batch):
        ''' Encoder: just encoding node features '''
        task_name, batch = batch['task_name'], batch['data']
        task_index = self.tasks.index(task_name)
        input_hidden, randomness = self.encoders[task_name](batch)
        max_len = batch.length.max().item()
        hints = []
        output = None

        # Process for length
        hidden = input_hidden
        ''' Algorithm steps '''
        for step in range(max_len): 
            ''' last hidden is from the last algorithm step'''
            last_hidden = hidden
            ''' For each step, we apply the message passing steps '''
            for message_passing_step in range(self.cfg.MODEL.MSG_PASSING_STEPS): 
                ''' Process: one GNN layer; hidden is updated very message passing step '''
                hidden = self.processor[message_passing_step](input_hidden, hidden, last_hidden, randomness=randomness[:, step] if randomness is not None else None, 
                                                              edge_index=batch.edge_index, batch_assignment=batch.batch, task_index=task_index,
                                                              **{self.edge_weight_name: self.process_weights(batch) for _ in range(1) if (hasattr(batch, 'weights') and hasattr(self, "edge_weight_name")) })
                if self.cfg.MODEL.GRU.ENABLE:
                    hidden = self.gru(hidden, last_hidden)
            if self.training and self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT > 0.0:
                ''' Decoder: just decoding node features to corresponding types '''
                ''' Decode for every algorithmic step '''
                hints.append(self.decoders[task_name](stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.DECODER_USE_LAST_HIDDEN), batch, 'hints'))

            # Check if output needs to be constructed
            if (batch.length == step+1).sum() > 0:
                # Decode outputs
                if self.training and self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT > 0.0:
                    # The last hint is the output, no need to decode again, its the same decoder
                    output_step = grab_outputs(hints[-1], batch)
                else:
                    output_step = self.decoders[task_name](stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.DECODER_USE_LAST_HIDDEN), batch, 'outputs')
                
                # Mask output
                mask = output_mask(batch, step)   
                if output is None:
                    output = {k: output_step[k]*mask[k] for k in output_step}
                else:
                    for k in output_step:
                        output[k][mask[k]] = output_step[k][mask[k]]

        hints = stack_hints(hints)

        return output, hints, hidden


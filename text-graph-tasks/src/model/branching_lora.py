"""
Tree-based Branching LoRA Module

This module implements a tree-structured branching LoRA adapter system where:
- Each layer can have multiple LoRA adapters (branches)
- Each task is routed to a specific adapter at each layer based on a configuration
- The routing creates a tree structure where tasks can share adapters at some layers
  and diverge at others
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import json
from collections import defaultdict
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
import copy


class BranchingLoraLayer(nn.Module):
    """
    A layer that contains multiple LoRA adapters (branches).
    Routes different tasks to different adapters based on configuration.
    """
    
    def __init__(self, base_layer, num_branches: int, lora_config: LoraConfig, branch_names: Optional[List[str]] = None):
        """
        Args:
            base_layer: The original layer to add LoRA branches to
            num_branches: Number of LoRA branches to create
            lora_config: LoRA configuration for each branch
            branch_names: Optional names for each branch (defaults to "branch_0", "branch_1", etc.)
        """
        super().__init__()
        self.base_layer = base_layer
        self.num_branches = num_branches
        self.lora_config = lora_config
        
        if branch_names is None:
            self.branch_names = [f"branch_{i}" for i in range(num_branches)]
        else:
            assert len(branch_names) == num_branches
            self.branch_names = branch_names
        
        # Create multiple LoRA branches
        self.lora_branches = nn.ModuleDict()
        self.branch_scalings = {}  # Store scaling factors separately (not nn.Modules)
        
        for branch_name in self.branch_names:
            # Create LoRA A and B matrices for this branch
            r = lora_config.r
            lora_alpha = lora_config.lora_alpha
            lora_dropout = lora_config.lora_dropout
            
            # Get input and output dimensions from base layer
            if hasattr(base_layer, 'in_features'):  # Linear layer
                in_features = base_layer.in_features
                out_features = base_layer.out_features
            elif hasattr(base_layer, 'in_channels'):  # Conv layer
                in_features = base_layer.in_channels
                out_features = base_layer.out_channels
            else:
                raise ValueError(f"Unsupported layer type: {type(base_layer)}")
            
            # Create LoRA matrices (following PEFT implementation)
            lora_A = nn.Linear(in_features, r, bias=False)
            lora_B = nn.Linear(r, out_features, bias=False)
            
            # Initialize as in PEFT
            nn.init.kaiming_uniform_(lora_A.weight, a=5**0.5)
            nn.init.zeros_(lora_B.weight)
            
            # Store scaling factor (not as a parameter, just as a value)
            scaling = lora_alpha / r
            self.branch_scalings[branch_name] = scaling
            
            # Create dropout if specified
            dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
            
            self.lora_branches[branch_name] = nn.ModuleDict({
                'lora_A': lora_A,
                'lora_B': lora_B,
                'dropout': dropout
            })
    
    def forward(self, x, branch_idx: Optional[int] = None):
        """
        Forward pass using a specific branch.
        
        Args:
            x: Input tensor
            branch_idx: Index of the branch to use (if None, uses branch 0)
        
        Returns:
            Output tensor with LoRA adaptation applied
        """
        if branch_idx is None:
            branch_idx = 0
            
        # Base layer output
        result = self.base_layer(x)
        
        # Add LoRA adaptation from selected branch
        branch_name = self.branch_names[branch_idx]
        branch = self.lora_branches[branch_name]
        scaling = self.branch_scalings[branch_name]
        
        lora_output = branch['lora_B'](branch['dropout'](branch['lora_A'](x)))
        result = result + lora_output * scaling
        
        return result
    
    def get_branch_idx(self, branch_name: str) -> int:
        """Get branch index from branch name."""
        return self.branch_names.index(branch_name)


class BranchingLoraModel(nn.Module):
    """
    Wraps a model with branching LoRA adapters.
    Manages task-to-branch routing across all layers.
    """
    
    def __init__(
        self, 
        base_model: nn.Module,
        branching_config: Dict,
        lora_config: LoraConfig,
        target_modules: List[str]
    ):
        """
        Args:
            base_model: The base model to add branching LoRA to
            branching_config: Configuration specifying task routing
                Format: {
                    "layer_0": {
                        "num_branches": 2,
                        "branch_names": ["branch_a", "branch_b"],
                        "task_to_branch": {"task1": "branch_a", "task2": "branch_b"}
                    },
                    ...
                }
            lora_config: LoRA configuration
            target_modules: List of module names to apply LoRA to (e.g., ["q_proj", "v_proj"])
        """
        super().__init__()
        self.base_model = base_model
        self.branching_config = branching_config
        self.lora_config = lora_config
        self.target_modules = target_modules
        self.current_task = None  # Track current task for routing
        
        # Build task-to-branch mapping for quick lookup
        self.task_routing = self._build_task_routing()
        
        # Map to store (module_name -> layer_idx) for routing
        self.module_to_layer = {}
        
        # Replace target modules with branching LoRA layers
        self._inject_branching_lora()
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Only LoRA parameters are trainable
        for name, module in self.named_modules():
            if isinstance(module, BranchingLoraLayer):
                for param in module.lora_branches.parameters():
                    param.requires_grad = True
    
    def __getattr__(self, name):
        """Delegate attribute access to base_model if not found in BranchingLoraModel."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to base_model
            return getattr(self.base_model, name)
    
    def _build_task_routing(self) -> Dict[str, Dict[int, str]]:
        """
        Build a mapping from task names to their branch at each layer.
        
        Returns:
            Dict mapping task_name -> layer_idx -> branch_name
        """
        task_routing = defaultdict(dict)
        
        for layer_key, layer_config in self.branching_config.items():
            # Extract layer index from key (e.g., "layer_0" -> 0)
            if isinstance(layer_key, str) and layer_key.startswith("layer_"):
                layer_idx = int(layer_key.split("_")[1])
            else:
                layer_idx = int(layer_key)
            
            task_to_branch = layer_config.get("task_to_branch", {})
            for task_name, branch_name in task_to_branch.items():
                task_routing[task_name][layer_idx] = branch_name
        
        return dict(task_routing)
    
    def _inject_branching_lora(self):
        """Replace target modules with branching LoRA layers."""
        def get_layer_index(name: str) -> int:
            """Extract layer index from module name."""
            import re
            patterns = [
                re.compile(r"(?:^|\.)(layers)\.(\d+)(?:\.|$)"),  # LLaMA-style
                re.compile(r"(?:^|\.)(h)\.(\d+)(?:\.|$)"),       # GPT-2-style
                re.compile(r"(?:^|\.)(layer)\.(\d+)(?:\.|$)"),   # BERT-style
            ]
            for pat in patterns:
                m = pat.search(name)
                if m:
                    return int(m.group(2))
            return 0
        
        # Find all modules to replace
        modules_to_replace = []
        for name, module in self.base_model.named_modules():
            # Check if this module should have LoRA
            if any(target in name for target in self.target_modules):
                layer_idx = get_layer_index(name)
                layer_key = f"layer_{layer_idx}"
                
                # Check if we have branching config for this layer
                if layer_key in self.branching_config:
                    modules_to_replace.append((name, module, layer_idx, layer_key))
        
        # Replace modules
        for name, module, layer_idx, layer_key in modules_to_replace:
            layer_config = self.branching_config[layer_key]
            num_branches = layer_config.get("num_branches", 1)
            branch_names = layer_config.get("branch_names", None)
            
            # Create branching LoRA layer
            branching_layer = BranchingLoraLayer(
                base_layer=module,
                num_branches=num_branches,
                lora_config=self.lora_config,
                branch_names=branch_names
            )
            
            # Store module to layer mapping
            self.module_to_layer[name] = layer_idx
            
            # Replace the module
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name:
                parent_module = self.base_model.get_submodule(parent_name)
            else:
                parent_module = self.base_model
            
            setattr(parent_module, child_name, branching_layer)
    
    def forward(self, *args, task_name: Optional[str] = None, **kwargs):
        """
        Forward pass with task-specific routing.
        
        Args:
            task_name: Name of the task (determines which branches to use)
            *args, **kwargs: Arguments passed to base model
        """
        if task_name is None:
            raise ValueError("task_name must be provided for branching LoRA forward pass")
        
        # Set the current task for routing
        self.current_task = task_name
        
        # Register forward hooks to route through correct branches
        hooks = []
        task_routing = self.task_routing.get(task_name, {})
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, BranchingLoraLayer):
                layer_idx = self.module_to_layer.get(name, 0)
                branch_name = task_routing.get(layer_idx, module.branch_names[0])
                branch_idx = module.get_branch_idx(branch_name)
                
                # Create a hook that injects branch_idx
                def make_hook(b_idx):
                    def hook(mod, inp):
                        # Return modified input with branch_idx
                        if isinstance(inp, tuple):
                            return inp + (b_idx,)
                        return (inp, b_idx)
                    return hook
                
                handle = module.register_forward_pre_hook(make_hook(branch_idx))
                hooks.append(handle)
        
        try:
            # Forward through base model
            outputs = self.base_model(*args, **kwargs)
        finally:
            # Remove hooks
            for handle in hooks:
                handle.remove()
        
        return outputs
    
    def generate(self, *args, task_name: Optional[str] = None, **kwargs):
        """
        Generate method with task-specific routing.
        
        This wraps the base model's generate method and handles branch routing.
        """
        if task_name is None:
            # If no task_name, try to use the last set task or raise error
            if self.current_task is None:
                raise ValueError("task_name must be provided for branching LoRA generate")
            task_name = self.current_task
        
        # Set up routing hooks
        hooks = []
        task_routing = self.task_routing.get(task_name, {})
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, BranchingLoraLayer):
                layer_idx = self.module_to_layer.get(name, 0)
                branch_name = task_routing.get(layer_idx, module.branch_names[0])
                branch_idx = module.get_branch_idx(branch_name)
                
                def make_hook(b_idx):
                    def hook(mod, inp):
                        if isinstance(inp, tuple):
                            return inp + (b_idx,)
                        return (inp, b_idx)
                    return hook
                
                handle = module.register_forward_pre_hook(make_hook(branch_idx))
                hooks.append(handle)
        
        try:
            # Call base model's generate
            outputs = self.base_model.generate(*args, **kwargs)
        finally:
            # Remove hooks
            for handle in hooks:
                handle.remove()
        
        return outputs
    
    @property
    def config(self):
        """Expose base model's config."""
        return self.base_model.config
    
    def save_branch_weights(self, save_path: str, branch_names: Optional[List[str]] = None):
        """
        Save weights for specific branches.
        
        Args:
            save_path: Path to save weights
            branch_names: List of branch names to save (None = save all)
        """
        state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, BranchingLoraLayer):
                for branch_name, branch in module.lora_branches.items():
                    if branch_names is None or branch_name in branch_names:
                        prefix = f"{name}.lora_branches.{branch_name}"
                        for param_name, param in branch.named_parameters():
                            state_dict[f"{prefix}.{param_name}"] = param.data
        
        torch.save(state_dict, save_path)
    
    def load_branch_weights(self, load_path: str, strict: bool = False):
        """Load branch weights from a checkpoint."""
        state_dict = torch.load(load_path)
        self.load_state_dict(state_dict, strict=strict)
    
    def get_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.2f}%"
        )


def load_branching_config(config_path: str) -> Dict:
    """
    Load branching configuration from a JSON file.
    
    Expected format:
    {
        "layer_0": {
            "num_branches": 2,
            "branch_names": ["branch_a", "branch_b"],
            "task_to_branch": {
                "task1": "branch_a",
                "task2": "branch_b"
            }
        },
        "layer_1": {
            "num_branches": 3,
            "branch_names": ["branch_x", "branch_y", "branch_z"],
            "task_to_branch": {
                "task1": "branch_x",
                "task2": "branch_y",
                "task3": "branch_z"
            }
        }
    }
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_branching_lora_model(
    base_model: nn.Module,
    branching_config_path: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
) -> BranchingLoraModel:
    """
    Convenience function to create a branching LoRA model.
    
    Args:
        base_model: Base language model
        branching_config_path: Path to branching configuration JSON
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_modules: Target modules for LoRA (auto-detected if None)
    
    Returns:
        BranchingLoraModel instance
    """
    # Load branching config
    branching_config = load_branching_config(branching_config_path)
    
    # Auto-detect target modules if not provided
    if target_modules is None:
        # Check model architecture
        model_type = base_model.config.model_type if hasattr(base_model, 'config') else None
        if model_type in ['llama', 'mistral', 'qwen2']:
            target_modules = ['q_proj', 'k_proj', 'v_proj']
        elif model_type == 'gpt2':
            target_modules = ['c_attn', 'c_proj']
        else:
            # Default to common attention projections
            target_modules = ['q_proj', 'k_proj', 'v_proj']
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Create branching LoRA model
    model = BranchingLoraModel(
        base_model=base_model,
        branching_config=branching_config,
        lora_config=lora_config,
        target_modules=target_modules
    )
    
    return model

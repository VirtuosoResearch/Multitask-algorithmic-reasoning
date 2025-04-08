import argparse
import logging
import os
import wandb

from src.custom.algorithm_task_data_module import AlgorithmDataModule
from src.custom.algorithm_task_graph_data_module import AlgorithmGraphDataModule
from src.model.GraphLlama_Graphqa import GraphLlamaForCausalLM_GraphQA
from src.model.qwen2_mixup import Qwen2MixupForCausalLM
from src.custom.multitask_model_graphqa import MultitaskModel_GraphQA
from src.custom.multitask_model import MultitaskModel

from functools import partial
from pytorch_lightning.trainer.states import RunningStage, TrainerFn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.nn import Embedding

from peft import get_peft_model, LoraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from collections import defaultdict
import time
import re

from adapters import SeqBnInvConfig, PrefixTuningConfig, BnConfig, DoubleSeqBnConfig, SeqBnConfig
from adapters import AutoAdapterModel,list_adapters, BnConfig
from torch._inductor.async_compile import AsyncCompile

logging.basicConfig(filename='log.log', level=logging.INFO, force=True, filemode='a')
torch.set_float32_matmul_precision("high")

# Import branching LoRA module
from src.model.branching_lora import BranchingLoraModel, create_branching_lora_model

# peft.__version__ '0.12.0'    

def add_result_to_csv(result_datapoint, file_name):
    for key, val in result_datapoint.items():
        result_datapoint[key] = [val, ]
    
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        tmp_df = pd.DataFrame(result_datapoint)
        result_df = pd.concat([result_df, tmp_df], ignore_index = True)
        result_df.to_csv(file_name)
    else:
        result_df = pd.DataFrame(result_datapoint)  
        result_df.to_csv(file_name)   

def layer_index_from_name(name, default=0):
    _LAYER_PATTERNS = [
        re.compile(r"(?:^|\.)(layers)\.(\d+)(?:\.|$)"),  # LLaMA-style: ...layers.{idx}....
        re.compile(r"(?:^|\.)(h)\.(\d+)(?:\.|$)"),       # GPT-2/NeoX-style: ...h.{idx}....
        re.compile(r"(?:^|\.)(layer)\.(\d+)(?:\.|$)"),   # BERT-style: ...layer.{idx}....
    ]
    for pat in _LAYER_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(2))
    return default

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
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": args.devices[0]}) #
        elif args.use_graph_llama:
            model = GraphLlamaForCausalLM_GraphQA.from_pretrained(hf_key)
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_key)
        model_type = "decoder"
        append_eos = True
    elif "flan" in model_key:
        hf_key = "google/{}".format(model_key.replace("_", "-"))
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False  # t5 tokenizers already append eos
    elif "Qwen" in model_key:
        hf_key = args.model_key.replace("_", "-")
        if args.train_invariant_mix:
            model = Qwen2MixupForCausalLM.from_pretrained(hf_key)
            model.set_alpha(args.invariant_mix_alpha)
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "decoder"
        append_eos = True
    else:
        raise NotImplementedError(args.model_key)
    
    if args.use_graph_llama:
        model.get_model().init_gnn()
        
        model.requires_grad_(True)
        for p in model.get_model().get_graph_tower().parameters():
            p.requires_grad = True
        for p in model.get_model().graph_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().graph_token_embeddings.parameters():
            p.requires_grad = True
        
        # # only the graph tower is trainable
        # if not args.freeze_graph_tower:
        #     for p in model.get_model().get_graph_tower().parameters():
        #         p.requires_grad = True
        # for p in model.get_model().graph_projector.parameters():
        #     p.requires_grad = True
        # if args.use_cross_attn:
        #     if not args.freeze_embeddings:
        #         for p in model.get_model().graph_token_embeddings.parameters():
        #             p.requires_grad = True
        #     for p in model.get_model().graph_token_cross_attn.parameters():
        #         p.requires_grad = True
        # model.set_if_only_train_graph(args.only_train_graph)
        
    if args.train_adapter:
        
        if args.use_qadapter:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4' 
            )

            model = AutoAdapterModel.from_pretrained(
                hf_key, 
                quantization_config=quantization_config, 
                torch_dtype=torch.bfloat16, 
                device_map={"": args.devices[0]}
            )
        
        else: model = AutoAdapterModel.from_pretrained(hf_key)

        bottleneck_config = DoubleSeqBnConfig(
            mh_adapter=True,    
            output_adapter=True,    
            reduction_factor=args.reduction_factor,     
            non_linearity="relu"     
        )

        model.add_adapter(adapter_name="seq_bn",config=bottleneck_config)

        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

        model.set_active_adapters("seq_bn")
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params_count = sum(p.numel() for p in model.parameters())

        print(f"Trainable parameters: {trainable_params_count} || All parameters: {all_params_count} || ratio: {trainable_params_count/all_params_count}")
        print("-"*20,"Bottleneck_Adapter","-"*20)

    
    if args.use_3bit or args.use_2bit:
        ''' deprecated '''
        from src.lqlora_utils import lora_utils
        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            use_gradient_checkpointing=True)

        lora_utils.transform_lora_layers(
            lpq=False,
            model=model,
            model_name="nf3" if args.use_3bit else "nf2",
            device=f"cuda:{args.devices[0]}")
        model.to(f"cuda:{args.devices[0]}")        

    elif args.train_lora:
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
        
    elif args.train_branching_lora:
        if args.branching_lora_config is None:
            raise ValueError("--branching_lora_config must be specified when using --train_branching_lora")
        
        # Determine target modules based on model architecture
        if args.model_key == "gpt2":
            target_modules = ["c_attn", "c_proj", "c_fc"]
        elif args.model_key == "EleutherAI/gpt-neox-20b":
            target_modules = ["query_key_value"]
        elif "flan" in args.model_key:
            target_modules = ["q", "k", "v"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj"]
        
        # Create branching LoRA model
        model = create_branching_lora_model(
            base_model=model,
            branching_config_path=args.branching_lora_config,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        
        model.print_trainable_parameters()
        print("-" * 20, "Branching LoRA", "-" * 20)

    return model, tokenizer, hf_key, model_type, append_eos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 'node_degree', 'node_count', 'edge_count', 'connected_nodes', 'cycle_check', 'disconnected_nodes', 'reachability', 'shortest_path', 'maximum_flow', 'triangle_counting', 'node_classification'
    parser.add_argument("--task_names", type=str, nargs="+", default=['edge_existence']) 
    parser.add_argument("--graph_types", type=str, nargs="+", default=['er'])
    # "adjacency" "incident" "friendship" "south_park" "got" "politician" "social_network" "expert" "coauthorship" "random" 
    parser.add_argument("--text_encoders", type=str, nargs="+", default=['adjacency']) 
    parser.add_argument("--min_nodes", type=int, default=20)
    parser.add_argument("--max_nodes", type=int, default=30)

    parser.add_argument("--eval_task_names", type=str, nargs="+", default=None)
    parser.add_argument("--eval_graph_types", type=str, nargs="+", default=None) 
    parser.add_argument("--eval_text_encoders", type=str, nargs="+", default=None)
    parser.add_argument("--eval_min_nodes", type=int, nargs="+", default=[20])
    parser.add_argument("--eval_max_nodes", type=int, nargs="+", default=[30])
    parser.add_argument("--eval_max_length", type=int, default=2048)
    parser.add_argument("--eval_max_output_length", type=int, default=64)

    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--task_idxes", type=int, nargs="+", default=None)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--downsample_ratio", type=float, default=1.0)
    parser.add_argument("--minimum_samples", type=int, default=1e6)
    parser.add_argument("--minimum_samples_validation", type=int, default=1e6)

    parser.add_argument("--train_adapter", action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=128)
    parser.add_argument("--use_qadapter", action="store_true")
    
    parser.add_argument("--train_branching_lora", action="store_true")
    parser.add_argument("--branching_lora_config", type=str, default=None, help="Path to branching LoRA config JSON file")

    parser.add_argument("--use_graph_llama", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_3bit", action="store_true")
    parser.add_argument("--use_2bit", action="store_true")

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--load_model_dir", type=str, default="test")
    parser.add_argument("--load_branching_config", type=str)
    parser.add_argument("--task_branching_config_dir", type=str)

    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--generate_output", action="store_true")
    parser.add_argument("--remove_checkpoint", action="store_true")

    parser.add_argument("--train_invariant_mix", action="store_true")
    parser.add_argument("--invariant_mix_alpha", type=float, default=0.1)

    args = parser.parse_args()
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    model_key = args.model_key.replace("/", "-").replace("..", "")
    save_name = model_key + "_" + \
                ("_".join(args.task_names) if len("_".join(args.task_names)) <= 100 else "{}_tasks".format(len(args.task_names))) + \
                ("_downsample_ratio_{}".format(args.downsample_ratio)) + \
                (f"_{args.save_name}" if args.save_name else "") + \
                (f"_lr_{args.lr}_wd_{args.weight_decay}") + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                (f"_lora_a_{args.lora_alpha}" if args.train_lora else "") + \
                (f"_nodes_{args.min_nodes}_{args.max_nodes}")
    print("save_name:", save_name)
    file_dir = os.path.join("./results/", save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    metrics = {}
    for run in range(args.runs):
        model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_size = args.batch_size
        if args.inference_batch_size is None:
            inference_batch_size = batch_size
        else:
            inference_batch_size = args.inference_batch_size
        
        if args.use_graph_llama:
            data_module = AlgorithmGraphDataModule(
                task_names=args.task_names,
                graph_types=args.graph_types,
                text_encoders=args.text_encoders,
                node_range=[args.min_nodes, args.max_nodes],
                tokenizer=tokenizer,
                batch_size=batch_size,
                inference_batch_size=inference_batch_size,
                max_input_length=args.max_length,
                max_output_length=args.max_output_length,
                eval_all=True,
                downsample_ratio=args.downsample_ratio,
                minimum_samples=args.minimum_samples,
                minimum_samples_validation=args.minimum_samples_validation)
        else:
            data_module = AlgorithmDataModule(
                task_names=args.task_names,
                graph_types=args.graph_types,
                text_encoders=args.text_encoders,
                node_range=[args.min_nodes, args.max_nodes],
                tokenizer=tokenizer,
                batch_size=batch_size,
                inference_batch_size=inference_batch_size,
                max_input_length=args.max_length,
                max_output_length=args.max_output_length,
                eval_all=True,
                downsample_ratio=args.downsample_ratio,
                minimum_samples=args.minimum_samples,
                minimum_samples_validation=args.minimum_samples_validation)
        data_module.setup(stage="fit")

        extended_task_names = [f"{task_name}_{graph_type}" for task_name, graph_type in zip(args.task_names, args.graph_types)]
        model_cls = MultitaskModel # MultitaskModel_GraphQA if args.use_graph_llama else
        lm = model_cls(model, tokenizer, model_type, use_cpu_offload=False,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, max_output_length=args.max_output_length, use_wandb=args.use_wandb, 
                        optimizer=args.optimizer, generate_output=args.generate_output, task_names=extended_task_names,
                        train_invariant_mix=args.train_invariant_mix)
        
        load_model_dir = args.load_model_dir
        load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
        if load_model_dir is not None:
            if ("ckpt" in load_model_dir) and os.path.exists(load_model_dir):
                lm = model_cls.load_from_checkpoint(load_model_dir, model=model, tokenizer=tokenizer, model_type=model_type,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, max_output_length=args.max_output_length, use_wandb=args.use_wandb,
                        optimizer=args.optimizer, generate_output=args.generate_output, task_names=extended_task_names)
                print(f"Loaded model from {load_model_dir}")
            elif ("pt" in load_model_dir) and os.path.exists(load_model_dir):
                model.load_state_dict(torch.load(load_model_dir), strict=False)
                print(f"Loaded model from {load_model_dir}")
        if args.load_branching_config:
            # load weights from different trained adapters
            if args.task_branching_config_dir is not None:
                task_branching_config_dir = os.path.join("branching_configs", args.task_branching_config_dir)
                with open(task_branching_config_dir, "r") as f:
                    for line in f.readlines():
                        layers, checkpoint_dir = line.split(":")
                        layers = layers.split(",")
                        checkpoint = torch.load(checkpoint_dir)
                        new_checkpoint = {}
                        for key, val in checkpoint.items():
                            if str(layer_index_from_name(key)) in layers:
                                new_checkpoint[key] = val
                        model.load_state_dict(new_checkpoint, strict=False)

        if not os.path.exists("external_lightning_logs"):
            raise Exception("external_lightning_logs/ does not exist")
        default_root_dir = os.path.join("external_lightning_logs", 
                                        f"{model_key}_" + \
                                        ("_".join(extended_task_names) if len("_".join(extended_task_names)) <= 100 else "{}_tasks".format(len(extended_task_names))) + \
                                        (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                                        (f"_{args.save_name}" if args.save_name else "") + \
                                        f"_run_{run}"
                                        )
        # # remove previous checkpoints
        # if args.save_name and os.path.exists(default_root_dir):
        #     os.system(f"rm -rf {default_root_dir}")
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=default_root_dir,
            filename="epoch_{epoch}",
            save_top_k=1,
            monitor='accuracy_score',
            save_last=True,
            mode="max",
        )

        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                            default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                            accumulate_grad_batches=args.accumulate, precision=args.precision,
                            enable_checkpointing=args.enable_checkpointing,
                            callbacks=[checkpoint_callback]
                            )
        # save initial weights
        if args.train_lora:
            if not os.path.exists(default_root_dir):
                os.makedirs(default_root_dir)
            model_path = default_root_dir + "/initial_weights.pt"
            state_dict = model.state_dict()
            state_dict = {k: v.clone() for k, v in state_dict.items() if "lora" in k}
            torch.save(state_dict, model_path)
        elif args.train_branching_lora:
            if not os.path.exists(default_root_dir):
                os.makedirs(default_root_dir)
            model_path = default_root_dir + "/initial_weights.pt"
            state_dict = model.state_dict()
            # Save all branching LoRA parameters
            state_dict = {k: v.clone() for k, v in state_dict.items() if "lora_branches" in k}
            torch.save(state_dict, model_path)

        start_time = time.time()
        if args.epochs > 0:
            #print(a)
            lm.train()
            trainer.fit(lm, datamodule=data_module)
            if args.train_lora:
                from lightning_fabric.utilities.cloud_io import _load as pl_load
                checkpoint = pl_load(checkpoint_callback.best_model_path, map_location=lm.device)
                state_dict = checkpoint["state_dict"]
                state_dict = {k[6:]: v for k, v in state_dict.items() if "lora" in k}
                torch.save(state_dict, checkpoint_callback.best_model_path.replace(".ckpt", ".pt"))
            elif args.train_branching_lora:
                from lightning_fabric.utilities.cloud_io import _load as pl_load
                checkpoint = pl_load(checkpoint_callback.best_model_path, map_location=lm.device)
                state_dict = checkpoint["state_dict"]
                state_dict = {k[6:]: v for k, v in state_dict.items() if "lora_branches" in k}
                torch.save(state_dict, checkpoint_callback.best_model_path.replace(".ckpt", ".pt"))
            elif args.train_adapter:
                from lightning_fabric.utilities.cloud_io import _load as pl_load
                checkpoint = pl_load(checkpoint_callback.best_model_path, map_location=lm.device)
                state_dict = checkpoint["state_dict"]
                state_dict = {k[6:]: v for k, v in state_dict.items() if ("adapter" in k or "head" in k)}
                torch.save(state_dict, checkpoint_callback.best_model_path.replace(".ckpt", ".pt"))
        end_time = time.time()
        print(f"Training time: {end_time - start_time}")

        # evaluate the best checkpoint
        start_time = time.time()
            
        if args.epochs > 0:
            if  args.train_lora or args.train_branching_lora or args.train_adapter or \
                args.use_qadapter or args.use_qlora or \
                args.use_3bit or args.use_2bit:                        
                model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)
                model.load_state_dict(state_dict, strict=False)
                lm = model_cls(model, tokenizer, model_type, use_cpu_offload=False,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, max_output_length=args.max_output_length, use_wandb=args.use_wandb,
                        optimizer=args.optimizer, generate_output=args.generate_output, task_names=extended_task_names)
                if args.use_3bit or args.use_2bit:
                    trainer.validate_loop.trainer_fn = TrainerFn.FITTING
                    trainer.validate_loop.inference_mode = False
                summary = trainer.validate(lm, datamodule=data_module)[0]
            else:
                summary = trainer.validate(lm, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)[0]
            logging.info(summary)
        else:
            if args.use_3bit or args.use_2bit:
                trainer.validate_loop.trainer_fn = TrainerFn.FITTING
                trainer.validate_loop.inference_mode = False
            summary = trainer.validate(lm, datamodule=data_module)[0]
            logging.info(summary)
        
        # evaluate the model on the evaluation tasks
        if args.eval_task_names is not None:
            for i, eval_task_name in enumerate(args.eval_task_names):
                eval_data_module = AlgorithmDataModule(
                task_names=[eval_task_name],
                graph_types=[args.eval_graph_types[i]],
                text_encoders=[args.eval_text_encoders[i]],
                node_range=[args.eval_min_nodes[i], args.eval_max_nodes[i]],
                tokenizer=tokenizer,
                batch_size=batch_size,
                inference_batch_size=inference_batch_size,
                max_input_length=args.eval_max_length,
                max_output_length=args.eval_max_output_length,
                eval_all=True,
                downsample_ratio=args.downsample_ratio,
                minimum_samples=args.minimum_samples,
                minimum_samples_validation=args.minimum_samples_validation)
                eval_data_module.setup(stage="fit")

                extended_eval_task_name = f"{eval_task_name}_{args.eval_graph_types[i]}"
                train_task_names = lm.task_names # replace with the current task names
                lm.task_names = [extended_eval_task_name]
                eval_summary = trainer.validate(lm, datamodule=eval_data_module)[0]
                lm.task_names = train_task_names
                logging.info(eval_summary)

                for key in eval_summary:
                    metric_key = f"eval_{key}_node_{args.eval_min_nodes[i]}_{args.eval_max_nodes[i]}"
                    if metric_key not in metrics:
                        metrics[metric_key] = []
                    metrics[metric_key].append(eval_summary[key])

        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time}")
            
        for key in summary:
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(summary[key])

        # delete the whole model checkpoint and only keep the lora parameters
        if args.train_lora or args.train_branching_lora or args.train_adapter:
            os.system(f"rm {checkpoint_callback.best_model_path}")
        
        if args.remove_checkpoint:
            if os.path.exists(default_root_dir):
                os.system(f"rm -r {default_root_dir}")
            else:
                print(f"Directory {default_root_dir} does not exist, skipping removal.")
    
    for key in metrics:
        print("{}: {:.4f} +/- {:.4f}".format(key, np.mean(metrics[key]), np.std(metrics[key])))
    
    # save indexes 
    if args.write_results:
        for task_name in extended_task_names:
            result_datapoint = {
                "Task name": task_name,
                "Trained with": " ".join(extended_task_names),
            }
            for key, val in metrics.items():
                if task_name in key:
                    tmp_key = key.replace(f"{task_name}_", "")
                    result_datapoint[tmp_key] = np.mean(val)
            file_name = os.path.join(file_dir, "results.csv")
            add_result_to_csv(result_datapoint, file_name)
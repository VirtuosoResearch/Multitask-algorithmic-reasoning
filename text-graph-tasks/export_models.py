# %%
import os
import torch
from lightning_fabric.utilities.cloud_io import _load as pl_load

load_model_dir = "Qwen-Qwen2.5-14B_12_tasks_lora_r_16_clrs_all_algorithms_run_0/epoch_epoch=6.ckpt"
load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
save_model_dir = load_model_dir.replace(".ckpt", ".pt")

checkpoint = pl_load(load_model_dir, map_location=f"cpu")
state_dict = checkpoint["state_dict"]
state_dict = {k[6:]: v for k, v in state_dict.items() if "lora" in k}


print(list((state_dict.keys())))
torch.save(state_dict, f"{save_model_dir}")

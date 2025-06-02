import os
import gc
import numpy as np
import pandas as pd
import argparse

def obtain_task_combination_results(args, combined_task_str):
    mtl_results = {}

    # load trained results
    model_key = args.model_key.replace("/", "-").replace("..", "")
    save_name = model_key + \
                (f"_{args.save_name}" if args.save_name else "") + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "")
    file_dir = os.path.join("./results/", save_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.join(file_dir, "results.csv")
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        sampled_tasks = result_df["Trained with"].values
    else:
        sampled_tasks = []

    # extended_combined_task_str = " ".join([f"{task_name}_zero_shot" for task_name in combined_task_str.split()])
    # test the subset has been trained 
    if_trained = False
    for sample_task in sampled_tasks:
        if sample_task == combined_task_str:
            if_trained = True
            break
    num_tasks = len(combined_task_str.split(" "))
    tasks_str = " ".join([task_name[:-10] for task_name in combined_task_str.split(" ")])

    if not if_trained:
        # Training script
        os.system("python train.py --task_names {}\
                    --prompt_styles {}\
                    --text_encoders {} \
                    --model_key {} \
                    --devices {} --batch_size {} --inference_batch_size {} --max_length {} --max_output_length {} --generate_output --runs {} --lr {} --epochs {}\
                    --train_lora --lora_rank {} --lora_alpha {} \
                    --save_name {} --precision 'bf16-true' --write_results --downsample_ratio {} --minimum_samples {}".format(
                tasks_str, " ".join(['zero_shot']*num_tasks), " ".join(['incident']*num_tasks),
                args.model_key, 
                " ".join([str(device) for device in args.devices]), args.batch_size, args.inference_batch_size, args.max_length, args.max_output_length, args.runs, args.lr, args.epochs,
                args.lora_rank, args.lora_alpha,
                args.save_name, args.downsample_ratio, args.minimum_samples
        ))
        gc.collect()

    # load again trained results
    result_df = pd.read_csv(file_name, index_col=0)
    sampled_tasks = result_df["Trained with"].values

    result_df = result_df[result_df["Trained with"] == combined_task_str]
    tasks = combined_task_str.split(" ")
    for tmp_task in tasks:
        tmp_target = result_df[result_df["Task name"] == tmp_task]["accuracy"].values[0]
        mtl_results[tmp_task] = tmp_target
    
    return mtl_results

def obtain_stl_result(args, task_name):
    stl_result = obtain_task_combination_results(args, str(task_name))
    return stl_result[task_name]

def main(args):
    task_names = args.task_names
    task_names = [task_name + "_zero_shot" for task_name in task_names]
    rng = np.random.default_rng(1024)
    task_names = rng.permutation(task_names)
    print(task_names)

    groups = []
    cur_task_performance = {}
    for task_name in task_names:
        # Train task_id with every current group
        tmp_mtl_results = {}
        for group_idx, group in enumerate(groups):
            tmp_mtl_results[group_idx] = {}
            combined_task_str = " ".join(group[:]+[task_name])
            combined_task_results = obtain_task_combination_results(args, combined_task_str)
            
            tmp_mtl_results[group_idx].update(combined_task_results)
            
            for other_task_name, val in cur_task_performance.items():
                if other_task_name not in tmp_mtl_results[group_idx]:
                    tmp_mtl_results[group_idx][other_task_name] = val

        # if len(groups) < group_num; We can add a new group that only has this task_id
        if len(groups) < args.group_num:
            tmp_mtl_results[len(groups)] = {}
            tmp_mtl_results[len(groups)][task_name] = obtain_stl_result(args, task_name)
            for other_task_name, val in cur_task_performance.items():
                tmp_mtl_results[len(groups)][other_task_name] = val
        
        # choose the one with the best average performance
        print(tmp_mtl_results)
        combined_task_to_results = []
        for group_idx in tmp_mtl_results.keys():
            avg_performance = np.mean(list(tmp_mtl_results[group_idx].values()))
            combined_task_to_results.append((group_idx, avg_performance))
        combined_task_to_results.sort(key=lambda x:x[1])
        selected_group_idx = combined_task_to_results[-1][0]

        if selected_group_idx == len(groups):
            groups.append([task_name])
        else:
            groups[selected_group_idx].append(task_name)

        # update cur_task_performance
        for update_id in tmp_mtl_results[selected_group_idx].keys():
            cur_task_performance[update_id] = tmp_mtl_results[selected_group_idx][update_id]
        print(groups)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_names", type=str, nargs="+", default=['edge_existence']) 
    # parser.add_argument("--prompt_styles", type=str, nargs="+", default=['zero_shot'])
    # parser.add_argument("--text_encoders", type=str, nargs="+", default=['adjacency']) 

    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
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

    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_3bit", action="store_true")
    parser.add_argument("--use_2bit", action="store_true")

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--load_model_dir", type=str, default="test")
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--generate_output", action="store_true")
    
    # Sample tasks
    parser.add_argument("--save_name", type=str, default="sampled_tasks")
    parser.add_argument("--group_num", type=int, default=3)
    args = parser.parse_args()
    main(args)
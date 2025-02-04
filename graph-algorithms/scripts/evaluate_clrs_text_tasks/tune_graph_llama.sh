# for lr in 5e-4 1e-4 5e-5 1e-5
# do
# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 64 --max_output_length 256 --generate_output --runs 1 --lr $lr \
#     --save_name tuning_stl --epochs 10 --use_graph_llama \
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15

# # python train_clrs_text.py --task_names bfs \
# #     --model_key "meta-llama/Llama-3.1-8B" \
# #     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr \
# #     --save_name tuning_stl --epochs 10 --use_graph_llama --train_lora --lora_rank 16 --lora_alpha 128 \
# #     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15
# done


for lr in 5e-4 1e-4 
do
# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl --epochs 10 --use_graph_llama \
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.2-1B_bfs_lora_r_16_tuning_stl_run_0/epoch_epoch=9.pt" --freeze_graph_tower

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl --epochs 10 --use_graph_llama \
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.2-1B_bfs_lora_r_16_tuning_stl_run_0/epoch_epoch=9.pt" --freeze_graph_tower --use_cross_attn

python train_clrs_text.py --task_names bfs \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
    --save_name tuning_stl_lora --epochs 10 --use_graph_llama --train_lora --lora_rank 16 --lora_alpha 128\
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
    --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt"

python train_clrs_text.py --task_names bfs \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 32 --generate_output --runs 1 --lr $lr\
    --save_name tuning_stl_lora --epochs 10 --use_graph_llama --train_lora --lora_rank 16 --lora_alpha 128\
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
    --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" --only_load_last_output

python train_clrs_text.py --task_names bfs \
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
    --save_name tuning_stl_lora --epochs 10 --use_graph_llama --train_lora --lora_rank 16 --lora_alpha 128\
    --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
    --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" --freeze_graph_tower 



# Train cross attn
# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl --epochs 3 --use_graph_llama \
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" --freeze_graph_tower --use_cross_attn --add_output_projection

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl --epochs 3 --use_graph_llama \
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" --freeze_graph_tower --use_cross_attn

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl --epochs 3 --use_graph_llama \
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" --freeze_graph_tower --use_cross_attn --freeze_embeddings

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl --epochs 3 --use_graph_llama --train_lora --lora_rank 16 --lora_alpha 128\
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" --freeze_graph_tower --use_cross_attn --add_output_projection

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl --epochs 3 --use_graph_llama --train_lora --lora_rank 16 --lora_alpha 128\
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" --freeze_graph_tower --use_cross_attn

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr $lr\
#     --save_name tuning_stl --epochs 3 --use_graph_llama --train_lora --lora_rank 16 --lora_alpha 128\
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" --freeze_graph_tower --use_cross_attn --freeze_embeddings

done


# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 1e-3\
#     --save_name test_pretraining --epochs 20 --use_graph_llama \
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --only_train_graph --test_classifier_before_cross_attn

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 1e-3\
#     --save_name test_pretraining --epochs 20 --use_graph_llama \
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --only_train_graph --test_classifier_before_cross_attn
# # Accuracy: 0.9333333802223206

# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.2-1B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 1e-4\
#     --save_name test_pretraining_cross_attn --epochs 20 --use_graph_llama \
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.2-1B_bfs_test_pretraining_run_0/epoch_epoch=18.pt" --only_train_graph --use_cross_attn --freeze_graph_tower
# # accuracy: 0.9380


# python train_clrs_text.py --task_names bfs \
#     --model_key "meta-llama/Llama-3.1-8B" \
#     --devices 0 --batch_size 4 --inference_batch_size 4 --max_length 32 --max_output_length 256 --generate_output --runs 1 --lr 1e-4\
#     --save_name tuning_stl_lora --epochs 10 --use_graph_llama --train_lora --lora_rank 16 --lora_alpha 128\
#     --few_shot_k 0 --downsample_ratio 0.05 --minimum_samples 1000 --minimum_samples_validation 100 --train_lengths 15 --test_lengths 15 \
#     --load_model_dir "meta-llama-Llama-3.1-8B_bfs_test_pretraining_run_0/epoch_epoch=17.pt" --freeze_graph_tower 
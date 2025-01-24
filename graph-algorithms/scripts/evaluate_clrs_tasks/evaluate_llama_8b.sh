python train_clrs_text.py --task_names "bellman_ford" --prompt_styles "cot_few_shot_10"\
    --model_key "meta-llama/Llama-3.1-8B" \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 13000 --max_output_length 1024 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.01 --minimum_samples 10 --minimum_samples_validation 32 --evaluate_cot\
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

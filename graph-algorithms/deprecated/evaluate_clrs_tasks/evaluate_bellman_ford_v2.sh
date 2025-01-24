for model in "meta-llama/Llama-3.2-1B"
do
python train_clrs_text.py --task_names "bfs" --prompt_styles "few_shot_10"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 12000 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 

python train_clrs_text.py --task_names "bfs" --prompt_styles "cot_few_shot_10"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 13000 --max_output_length 1024 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 --evaluate_cot 
done

for algorithm in "bellman_ford" "bfs"
do
for model in "meta-llama/Llama-3.1-8B" 
do
python train_clrs_text.py --task_names $algorithm --prompt_styles "zero_shot"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 2000 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32\
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

python train_clrs_text.py --task_names $algorithm --prompt_styles "few_shot_2"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 3500 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32\
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

python train_clrs_text.py --task_names $algorithm --prompt_styles "cot_few_shot_2"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 5000 --max_output_length 1024 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 --evaluate_cot \
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

python train_clrs_text.py --task_names $algorithm --prompt_styles "few_shot_4"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 5000 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

python train_clrs_text.py --task_names $algorithm --prompt_styles "cot_few_shot_4"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 7000 --max_output_length 1024 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 --evaluate_cot \
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

python train_clrs_text.py --task_names $algorithm --prompt_styles "few_shot_6"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 7000 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

python train_clrs_text.py --task_names $algorithm --prompt_styles "cot_few_shot_6"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 9000 --max_output_length 1024 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 --evaluate_cot \
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

python train_clrs_text.py --task_names $algorithm --prompt_styles "few_shot_8"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 9000 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

python train_clrs_text.py --task_names $algorithm --prompt_styles "cot_few_shot_8"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 11000 --max_output_length 1024 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 --evaluate_cot \
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

python train_clrs_text.py --task_names $algorithm --prompt_styles "few_shot_10"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 12000 --max_output_length 256 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 \
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora

python train_clrs_text.py --task_names $algorithm --prompt_styles "cot_few_shot_10"\
    --model_key $model \
    --devices 0 --batch_size 1 --inference_batch_size 1 --max_length 13000 --max_output_length 1024 --generate_output --runs 1 --lr 5e-5 \
    --save_name test --epochs 0 --precision "bf16-true" --evaluate_training_set --downsample 0.1 --minimum_samples 100 --minimum_samples_validation 32 --evaluate_cot \
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora
done
done

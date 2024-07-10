# python train_multitask.py --config configs/config.json \
#     --algorithms addition --data_dirs digit_5_carry_True \
#     --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
#     --lr 5e-3 --random_init --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_sizes 10000 --runs 1 --concatenate_steps

python train.py --config configs/config.json \
    --algorithms bfs --data_dirs bfs_data_10 \
    --batch_size 256 --max_length 150 --num_of_instances 500 --generate_length 128\
    --lr 5e-3 --random_init --device 0 --num_layers 2 --tokenizer_dir gpt2_bfs\
    --train_sizes 100 --eval_epoch 80 --early_stop 800 --runs 1
# python train_single_prompt.py --config configs/config.json \
#     --algorithms addition --data_dirs digit_5_carry_True \
#     --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
#     --lr 5e-3 --random_init --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_sizes 2000 --eval_epoch 80 --runs 1

# python train_single_prompt.py --config configs/config.json \
#     --algorithms addition --data_dirs digit_5_carry_True \
#     --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
#     --lr 5e-3 --random_init --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_sizes 5000 --eval_epoch 40 --runs 1

# python train_single_prompt.py --config configs/config.json \
#     --algorithms addition --data_dirs digit_5_carry_True \
#     --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
#     --lr 5e-3 --random_init --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_sizes 10000 --eval_epoch 20 --runs 1

# python train_single_prompt.py --config configs/config.json \
#     --algorithms addition --data_dirs digit_5_carry_True \
#     --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
#     --lr 5e-3 --random_init --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_sizes 20000 --eval_epoch 10 --runs 1

# python train_single_prompt.py --config configs/config.json \
#     --algorithms addition --data_dirs digit_5_carry_True \
#     --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
#     --lr 5e-3 --random_init --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
#     --train_sizes 50000 --eval_epoch 5 --runs 1
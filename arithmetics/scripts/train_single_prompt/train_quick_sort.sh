# python train_multitask.py --config configs/config.json \
#     --algorithms quick_sort --data_dirs inter_results_length_5 \
#     --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
#     --lr 5e-3 --random_init --device 3 --num_layers 2 --tokenizer_dir gpt2_sort_100\
#     --train_sizes 50000 --runs 1 --concatenate_steps

python train_single_prompt.py --config configs/config.json \
    --algorithms quick_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_sizes 2000 --eval_epoch 80 --early_stop 800 --runs 1

python train_single_prompt.py --config configs/config.json \
    --algorithms quick_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_sizes 5000 --eval_epoch 40 --runs 1

python train_single_prompt.py --config configs/config.json \
    --algorithms quick_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_sizes 10000 --eval_epoch 20 --runs 1

python train_single_prompt.py --config configs/config.json \
    --algorithms quick_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_sizes 20000 --eval_epoch 10 --runs 1

python train_single_prompt.py --config configs/config.json \
    --algorithms quick_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_sizes 50000 --eval_epoch 5 --runs 1

python train_single_prompt.py --config configs/config.json \
    --algorithms quick_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_sizes 100000 --eval_epoch 1 --runs 1
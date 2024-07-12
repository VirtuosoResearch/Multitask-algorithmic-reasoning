# for size in 50000 # 20000 10000
# do
# python train_multitask.py --config configs/config.json \
#     --algorithms insertion_sort --data_dirs inter_results_length_10 \
#     --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
#     --lr 5e-4 --random_init --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
#     --train_sizes $size --max_steps 100000 --runs 1
# done
 
python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-3 --random_init --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 100000 --max_steps 400000 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-3 --random_init --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 50000 --max_steps 200000 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-3 --random_init --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 20000 --max_steps 200000 --runs 1
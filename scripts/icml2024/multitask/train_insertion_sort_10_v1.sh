# for size in 5000 2000 1000
# do
# python train_multitask.py --config configs/config.json \
#     --algorithms insertion_sort --data_dirs inter_results_length_10 \
#     --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
#     --lr 5e-4 --random_init --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
#     --train_sizes $size --runs 1
# done

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-3 --random_init --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 1000 --eval_epoch 50 --max_steps 100000 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-3 --random_init --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 2000 --eval_epoch 20 --max_steps 100000 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-3 --random_init --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
   --train_sizes 5000 --eval_epoch 10  --max_steps 100000 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_10 \
    --batch_size 256 --max_length 32 --num_of_instances 1000000 --generate_length 10\
    --lr 5e-3 --random_init --device 3 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
   --train_sizes 10000 --eval_epoch 2  --max_steps 100000 --runs 1
python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --random_init --device 2 --num_layers 6 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 1000 --eval_epoch 50 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --random_init --device 2 --num_layers 6 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 2000 --eval_epoch 20 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --random_init --device 2 --num_layers 6 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 5000 --eval_epoch 10 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --random_init --device 2 --num_layers 6 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 10000 --eval_epoch 2 --runs 1

for size in 20000 50000
do
python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --random_init --device 2 --num_layers 6 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes $size --runs 1
done

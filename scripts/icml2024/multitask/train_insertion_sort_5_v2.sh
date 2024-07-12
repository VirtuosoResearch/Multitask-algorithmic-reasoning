python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --random_init --device 1 --num_layers 12 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 1000 --eval_epoch 50 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --random_init --device 1 --num_layers 12 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 2000 --eval_epoch 20 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --random_init --device 1 --num_layers 12 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 5000 --eval_epoch 10 --runs 1

python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --random_init --device 1 --num_layers 12 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes 10000 --eval_epoch 2 --runs 1

for size in 20000 50000
do
python train_multitask.py --config configs/config.json \
    --algorithms insertion_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-3 --random_init --device 1 --num_layers 12 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes $size --runs 1
done

# for size in 50000 #100000 # 1000 2000 5000 
# do
# python train_multitask.py --config configs/config.json \
#     --algorithms insertion_sort --data_dirs inter_results_length_5 \
#     --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
#     --lr 5e-4 --random_init --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
#     --train_sizes $size --runs 1
# done

# for lr in 1e-3 5e-3
# do
# python train_multitask.py --config configs/config.json \
#     --algorithms insertion_sort --data_dirs inter_results_length_5 \
#     --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
#     --lr $lr --random_init --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
#     --train_sizes 50000 --runs 1
# done

# for l in 6 12
# do
# python train_multitask.py --config configs/config.json \
#     --algorithms insertion_sort --data_dirs inter_results_length_5 \
#     --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
#     --lr 5e-4 --random_init --device 1 --num_layers $l --random_init --tokenizer_dir gpt2_sort_100\
#     --train_sizes 50000 --runs 1
# done

# python train_multitask.py --config configs/config.json \
#     --algorithms insertion_sort --data_dirs inter_results_length_5 \
#     --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
#     --lr 5e-3 --random_init --device 3 --num_layers 12 --random_init --tokenizer_dir gpt2_sort_100\
#     --train_sizes 50000 --runs 1

# python train_multitask.py --config configs/config.json \
#     --algorithms insertion_sort --data_dirs inter_results_length_5 \
#     --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
#     --lr 1e-2 --random_init --device 3 --num_layers 12 --random_init --tokenizer_dir gpt2_sort_100\
#     --train_sizes 50000 --runs 1

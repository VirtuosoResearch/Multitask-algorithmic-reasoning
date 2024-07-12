# for size in 1000 2000 5000
# do

# done

python train_multitask.py --config configs/config.json \
    --algorithms addition --data_dirs digit_10_carry_True \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_sizes 1000 --runs 1 --eval_epoch 50

python train_multitask.py --config configs/config.json \
    --algorithms addition --data_dirs digit_10_carry_True \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_sizes 2000 --runs 1 --eval_epoch 20

python train_multitask.py --config configs/config.json \
    --algorithms addition --data_dirs digit_10_carry_True \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_sizes 5000 --runs 1 --eval_epoch 2

python train_multitask.py --config configs/config.json \
    --algorithms addition --data_dirs digit_10_carry_True \
    --batch_size 256 --max_length 64 --num_of_instances 1000000 --generate_length 11\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_sizes 10000 --runs 1 --eval_epoch 1
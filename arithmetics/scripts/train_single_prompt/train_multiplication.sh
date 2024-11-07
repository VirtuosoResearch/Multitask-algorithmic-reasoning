python train_single_prompt.py --config configs/config.json \
    --algorithms multiplication --data_dirs digit_5_carry_True \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-4 --random_init --device 0 --num_layers 2 --tokenizer_dir gpt2_arithmetic\
    --train_sizes 1000 --eval_epoch 150 --early_stop 1200 --runs 1

python train_single_prompt.py --config configs/config.json \
    --algorithms multiplication --data_dirs digit_5_carry_True \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-4 --random_init --device 0 --num_layers 2 --tokenizer_dir gpt2_arithmetic\
    --train_sizes 2000 --eval_epoch 80 --early_stop 800 --runs 1

python train_single_prompt.py --config configs/config.json \
    --algorithms multiplication --data_dirs digit_5_carry_True \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-4 --random_init --device 0 --num_layers 2 --tokenizer_dir gpt2_arithmetic\
    --train_sizes 5000 --eval_epoch 40 --early_stop 200 --runs 1

python train_single_prompt.py --config configs/config.json \
    --algorithms multiplication --data_dirs digit_5_carry_True \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-4 --random_init --device 0 --num_layers 2 --tokenizer_dir gpt2_arithmetic\
    --train_sizes 10000 --eval_epoch 20 --runs 1
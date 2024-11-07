python train.py --config configs/config.json \
    --algorithms addition --data_dirs digit_5_carry_True \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_sizes 500 --eval_epoch 150 --early_stop 1200 --runs 1
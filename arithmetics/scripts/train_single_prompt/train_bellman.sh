python train.py --config configs/config.json \
    --algorithms bellman --data_dirs bellman_data_8_16 \
    --batch_size 256 --max_length 256 --num_of_instances 100000 --generate_length 256\
    --lr 5e-3 --random_init --device 0 --num_layers 2 --tokenizer_dir gpt2_bellman\
    --train_sizes 15000 --eval_epoch 80 --early_stop 800 --runs 1 --valid_size 1000 --test_size 1000

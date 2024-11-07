python train.py --config configs/config.json \
    --algorithms dfs --data_dirs dfs_first_data_20 \
    --batch_size 256 --max_length 128 --num_of_instances 10000 --generate_length 256\
    --lr 5e-3 --random_init --device 0 --num_layers 2  --tokenizer_dir gpt2_dfs\
    --train_sizes 5000 --eval_epoch 80 --runs 1 --valid_size 1000 --test_size 1000
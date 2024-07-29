python train.py --config configs/config.json \
    --algorithms bfs --data_dirs bfs_data_10 \
    --batch_size 256 --max_length 150 --num_of_instances 10000 --generate_length 512\
    --lr 5e-3 --random_init --device 0 --num_layers 2 --tokenizer_dir gpt2_bfs\
    --train_sizes 6000 --eval_epoch 80 --early_stop 800 --runs 1 --valid_size 500 --test_size 500
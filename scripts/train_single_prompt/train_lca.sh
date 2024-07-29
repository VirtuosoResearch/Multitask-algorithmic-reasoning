python train.py --config configs/config.json \
    --algorithms lca --data_dirs lca_data_10 \
    --batch_size 256 --max_length 256 --num_of_instances 100000 --generate_length 256\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_lca\
    --train_sizes 8000 --eval_epoch 80 --early_stop 800 --runs 1 --valid_size 1000 --test_size 1000

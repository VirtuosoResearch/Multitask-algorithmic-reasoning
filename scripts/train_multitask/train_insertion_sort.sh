python train_multiple_lengths.py --config configs/config.json \
    --algorithms insertion_sort insertion_sort --data_dirs inter_results_length_3 inter_results_length_5 \
    --batch_size 256 --max_length 128 --num_of_instances 1000000 --generate_length 128\
    --lr 5e-3 --random_init --device 1 --num_layers 2 --tokenizer_dir gpt2_sort_100\
    --train_sizes 2000 2000 --eval_epoch 1 --early_stop 800 --runs 1

for size in 50000 100000
do
python train_multitask.py --config configs/config.json \
    --algorithms quick_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --random_init --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes $size --runs 1
done
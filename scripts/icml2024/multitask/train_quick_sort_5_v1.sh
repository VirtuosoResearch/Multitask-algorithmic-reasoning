for size in 10000 20000 # 1000 2000 5000 
do
python train_multitask.py --config configs/config.json \
    --algorithms quick_sort --data_dirs inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --random_init --device 0 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_sizes $size --runs 1
done
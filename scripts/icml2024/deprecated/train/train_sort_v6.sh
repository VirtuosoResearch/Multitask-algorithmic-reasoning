for size in 2000 5000 10000 20000 50000 100000 200000 500000 1000000
do
python train.py --config configs/config.json \
    --algorithm heap_sort --data_dir inter_results_length_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 \
    --lr 5e-4 --random_init --device 2 --num_layers 2 --tokenizer_dir gpt2_sort_100\
     --train_size $size
done
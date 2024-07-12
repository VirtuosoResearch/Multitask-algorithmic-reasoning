for size in 10000 5000 2000  # 20000 50000 100000 200000 500000 1000000
do
python train.py --config configs/config.json \
    --algorithm sorting --data_dir length_80 \
    --batch_size 256 --max_length 256 --num_of_instances 1000000 --generate_length 80\
    --lr 5e-4 --device 1 --num_layers 2 --random_init --tokenizer_dir gpt2_sort_100\
    --train_size $size
done

for size in 1000 2000 5000 10000 20000 50000 #  100000 200000 500000 1000000
do
python train_multitask.py --config configs/config.json \
    --algorithms division --data_dirs digit_5 \
    --batch_size 256 --max_length 16 --num_of_instances 1000000 --generate_length 5\
    --lr 5e-4 --random_init --device 2 --num_layers 2 --random_init --tokenizer_dir gpt2_arithmetic\
    --train_sizes $size --runs 1
done

